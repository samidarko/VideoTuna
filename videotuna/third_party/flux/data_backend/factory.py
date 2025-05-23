import io
import json
import logging
import os
import queue
import threading
import time
from math import sqrt

import torch
from tqdm import tqdm

from videotuna.third_party.flux.caching.text_embeds import TextEmbeddingCache
from videotuna.third_party.flux.caching.vae import VAECache
from videotuna.third_party.flux.data_backend.aws import S3DataBackend
from videotuna.third_party.flux.data_backend.base import BaseDataBackend
from videotuna.third_party.flux.data_backend.csv_url_list import CSVDataBackend
from videotuna.third_party.flux.data_backend.local import LocalDataBackend
from videotuna.third_party.flux.multiaspect.dataset import MultiAspectDataset
from videotuna.third_party.flux.multiaspect.sampler import MultiAspectSampler
from videotuna.third_party.flux.prompts import PromptHandler
from videotuna.third_party.flux.training.collate import collate_fn
from videotuna.third_party.flux.training.default_settings import (
    default,
    latest_config_version,
)
from videotuna.third_party.flux.training.exceptions import MultiDatasetExhausted
from videotuna.third_party.flux.training.multi_process import _get_rank as get_rank
from videotuna.third_party.flux.training.multi_process import rank_info, should_log
from videotuna.third_party.flux.training.state_tracker import StateTracker

logger = logging.getLogger("DataBackendFactory")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)
prefetch_log = logging.getLogger("DataBackendPrefetch")
if should_log():
    prefetch_log.setLevel(os.environ.get("SIMPLETUNER_PREFETCH_LOG_LEVEL", "INFO"))
else:
    prefetch_log.setLevel(logging.ERROR)

# For prefetching.


def prefetch_log_debug(message):
    prefetch_log.debug(f"{rank_info()} {message}")


def info_log(message):
    if StateTracker.get_accelerator().is_main_process:
        logger.info(message)


def init_backend_config(backend: dict, args: dict, accelerator) -> dict:
    output = {"id": backend["id"], "config": {}}
    if backend.get("dataset_type", None) == "text_embeds":
        if "caption_filter_list" in backend:
            output["config"]["caption_filter_list"] = backend["caption_filter_list"]
        output["dataset_type"] = "text_embeds"

        return output
    elif backend.get("dataset_type", None) == "image_embeds":
        # no overrides for image embed backends
        return output
    else:
        ## Check for settings we shouldn't have for non-text datasets.
        if "caption_filter_list" in backend:
            raise ValueError(
                f"caption_filter_list is only a valid setting for text datasets. It is currently set for the {backend.get('dataset_type', 'image')} dataset {backend['id']}."
            )

    # Image backend config
    output["dataset_type"] = backend.get("dataset_type", "image")
    choices = ["image", "conditioning"]
    if (
        StateTracker.get_args().controlnet
        and output["dataset_type"] == "image"
        and backend.get("conditioning_data", None) is None
    ):
        raise ValueError(
            "Image datasets require a corresponding conditioning_data set configured in your dataloader."
        )
    if output["dataset_type"] not in choices:
        raise ValueError(f"(id={backend['id']}) dataset_type must be one of {choices}.")
    if "vae_cache_clear_each_epoch" in backend:
        output["config"]["vae_cache_clear_each_epoch"] = backend[
            "vae_cache_clear_each_epoch"
        ]
    if "probability" in backend:
        output["config"]["probability"] = (
            float(backend["probability"]) if backend["probability"] else 1.0
        )
    if "ignore_epochs" in backend:
        logger.error(
            "ignore_epochs is deprecated, and will do nothing. This can be safely removed from your configuration."
        )
    if "repeats" in backend:
        output["config"]["repeats"] = (
            int(backend["repeats"]) if backend["repeats"] else 0
        )
    if "crop" in backend:
        output["config"]["crop"] = backend["crop"]
    else:
        output["config"]["crop"] = False
    if backend.get("type") == "csv":
        if "csv_cache_dir" in backend:
            output["config"]["csv_cache_dir"] = backend["csv_cache_dir"]
        if "csv_file" in backend:
            output["config"]["csv_file"] = backend["csv_file"]
        if "csv_caption_column" in backend:
            output["config"]["csv_caption_column"] = backend["csv_caption_column"]
        if "csv_url_column" in backend:
            output["config"]["csv_url_column"] = backend["csv_url_column"]
    if "crop_aspect" in backend:
        choices = ["square", "preserve", "random", "closest"]
        if backend.get("crop_aspect", None) not in choices:
            raise ValueError(
                f"(id={backend['id']}) crop_aspect must be one of {choices}."
            )
        output["config"]["crop_aspect"] = backend["crop_aspect"]
        if (
            output["config"]["crop_aspect"] == "random"
            or output["config"]["crop_aspect"] == "closest"
        ):
            if "crop_aspect_buckets" not in backend or not isinstance(
                backend["crop_aspect_buckets"], list
            ):
                raise ValueError(
                    f"(id={backend['id']}) crop_aspect_buckets must be provided when crop_aspect is set to 'random'."
                    " This should be a list of float values or a list of dictionaries following the format: {'aspect_bucket': float, 'weight': float}."
                    " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
                )
            for bucket in backend.get("crop_aspect_buckets"):
                if type(bucket) not in [float, int, dict]:
                    raise ValueError(
                        f"(id={backend['id']}) crop_aspect_buckets must be a list of float values or a list of dictionaries following the format: {'aspect_bucket': float, 'weight': float}."
                        " The weight represents how likely this bucket is to be chosen, and all weights should add up to 1.0 collectively."
                    )

        output["config"]["crop_aspect_buckets"] = backend.get("crop_aspect_buckets")
    else:
        output["config"]["crop_aspect"] = "square"
    if "crop_style" in backend:
        crop_styles = ["random", "corner", "center", "centre", "face"]
        if backend["crop_style"] not in crop_styles:
            raise ValueError(
                f"(id={backend['id']}) crop_style must be one of {crop_styles}."
            )
        output["config"]["crop_style"] = backend["crop_style"]
    else:
        output["config"]["crop_style"] = "random"
    output["config"]["disable_validation"] = backend.get("disable_validation", False)
    if "resolution" in backend:
        output["config"]["resolution"] = backend["resolution"]
    else:
        output["config"]["resolution"] = args.resolution
    if "resolution_type" in backend:
        output["config"]["resolution_type"] = backend["resolution_type"]
    else:
        output["config"]["resolution_type"] = args.resolution_type
    if "parquet" in backend:
        output["config"]["parquet"] = backend["parquet"]
    if "caption_strategy" in backend:
        output["config"]["caption_strategy"] = backend["caption_strategy"]
    else:
        output["config"]["caption_strategy"] = args.caption_strategy
    output["config"]["instance_data_dir"] = backend.get(
        "instance_data_dir", backend.get("aws_data_prefix", "")
    )
    if "hash_filenames" in backend:
        output["config"]["hash_filenames"] = backend["hash_filenames"]
    if "hash_filenames" in backend and backend.get("type") == "csv":
        output["config"]["hash_filenames"] = backend["hash_filenames"]

    # check if caption_strategy=parquet with metadata_backend=json
    current_metadata_backend_type = backend.get("metadata_backend", "discovery")
    if output["config"]["caption_strategy"] == "parquet" and (
        current_metadata_backend_type == "json"
        or current_metadata_backend_type == "discovery"
    ):
        raise ValueError(
            f"(id={backend['id']}) Cannot use caption_strategy=parquet with metadata_backend={current_metadata_backend_type}. Instead, it is recommended to use the textfile strategy and extract your captions into txt files."
        )

    maximum_image_size = backend.get("maximum_image_size", args.maximum_image_size)
    target_downsample_size = backend.get(
        "target_downsample_size", args.target_downsample_size
    )
    output["config"]["maximum_image_size"] = maximum_image_size
    output["config"]["target_downsample_size"] = target_downsample_size

    if maximum_image_size and not target_downsample_size:
        raise ValueError(
            "When a data backend is configured to use `maximum_image_size`, you must also provide a value for `target_downsample_size`."
        )
    if (
        maximum_image_size
        and output["config"]["resolution_type"] == "area"
        and maximum_image_size > 10
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When a data backend is configured to use `'resolution_type':area`, `maximum_image_size` must be less than 10 megapixels. You may have accidentally entered {maximum_image_size} pixels, instead of megapixels."
        )
    elif (
        maximum_image_size
        and output["config"]["resolution_type"] == "pixel"
        and maximum_image_size < 512
        and "deepfloyd" not in args.model_type
        and args.model_family != "smoldit"
    ):
        raise ValueError(
            f"When a data backend is configured to use `'resolution_type':pixel`, `maximum_image_size` must be at least 512 pixels. You may have accidentally entered {maximum_image_size} megapixels, instead of pixels."
        )
    if (
        target_downsample_size
        and output["config"]["resolution_type"] == "area"
        and target_downsample_size > 10
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When a data backend is configured to use `'resolution_type':area`, `target_downsample_size` must be less than 10 megapixels. You may have accidentally entered {target_downsample_size} pixels, instead of megapixels."
        )
    elif (
        target_downsample_size
        and output["config"]["resolution_type"] == "pixel"
        and target_downsample_size < 512
        and "deepfloyd" not in args.model_type
        and args.model_family != "smoldit"
    ):
        raise ValueError(
            f"When a data backend is configured to use `'resolution_type':pixel`, `target_downsample_size` must be at least 512 pixels. You may have accidentally entered {target_downsample_size} megapixels, instead of pixels."
        )

    return output


def print_bucket_info(metadata_backend):
    # Print table header
    if get_rank() == 0:
        tqdm.write(f"{rank_info()} | {'Bucket':<10} | {'Image Count (per-GPU)':<12}")

        # Print separator
        tqdm.write("-" * 30)

        # Print each bucket's information
        for bucket in metadata_backend.aspect_ratio_bucket_indices:
            image_count = len(metadata_backend.aspect_ratio_bucket_indices[bucket])
            if image_count == 0:
                continue
            tqdm.write(f"{rank_info()} | {bucket:<10} | {image_count:<12}")


def configure_parquet_database(backend: dict, args, data_backend: BaseDataBackend):
    """When given a backend config dictionary, configure a parquet database."""
    parquet_config = backend.get("parquet", None)
    if not parquet_config:
        raise ValueError(
            "Parquet backend must have a 'parquet' field in the backend config containing required fields for configuration."
        )
    parquet_path = parquet_config.get("path", None)
    if not parquet_path:
        raise ValueError(
            "Parquet backend must have a 'path' field in the backend config under the 'parquet' key."
        )
    if not data_backend.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")
    # Load the dataframe
    import pandas as pd

    bytes_string = data_backend.read(parquet_path)
    pq = io.BytesIO(bytes_string)
    if parquet_path.endswith(".jsonl"):
        df = pd.read_json(pq, lines=True)
    else:
        df = pd.read_parquet(pq)

    caption_column = parquet_config.get(
        "caption_column", args.parquet_caption_column or "description"
    )
    fallback_caption_column = parquet_config.get("fallback_caption_column", None)
    filename_column = parquet_config.get(
        "filename_column", args.parquet_filename_column or "id"
    )
    identifier_includes_extension = parquet_config.get(
        "identifier_includes_extension", False
    )

    # Check the columns exist
    if caption_column not in df.columns:
        raise ValueError(
            f"Parquet file {parquet_path} does not contain a column named '{caption_column}'."
        )
    if filename_column not in df.columns:
        raise ValueError(
            f"Parquet file {parquet_path} does not contain a column named '{filename_column}'."
        )
    # Check for null values
    if df[caption_column].isnull().values.any() and not fallback_caption_column:
        raise ValueError(
            f"Parquet file {parquet_path} contains null values in the '{caption_column}' column, but no fallback_caption_column was set."
        )
    if df[filename_column].isnull().values.any():
        raise ValueError(
            f"Parquet file {parquet_path} contains null values in the '{filename_column}' column."
        )
    # Check for empty strings
    if (df[caption_column] == "").sum() > 0 and not fallback_caption_column:
        raise ValueError(
            f"Parquet file {parquet_path} contains empty strings in the '{caption_column}' column."
        )
    if (df[filename_column] == "").sum() > 0:
        raise ValueError(
            f"Parquet file {parquet_path} contains empty strings in the '{filename_column}' column."
        )
    # Store the database in StateTracker
    StateTracker.set_parquet_database(
        backend["id"],
        (
            df,
            filename_column,
            caption_column,
            fallback_caption_column,
            identifier_includes_extension,
        ),
    )
    info_log(
        f"Configured parquet database for backend {backend['id']}. Caption column: {caption_column}. Filename column: {filename_column}."
    )


def configure_multi_databackend(args: dict, accelerator, text_encoders, tokenizers):
    """
    Configure a multiple dataloaders based on the provided commandline args.
    """
    StateTracker.clear_data_backends()
    logger.setLevel(
        os.environ.get(
            "SIMPLETUNER_LOG_LEVEL", "INFO" if accelerator.is_main_process else "ERROR"
        )
    )
    if args.data_backend_config is None:
        raise ValueError(
            "Must provide a data backend config file via --data_backend_config"
        )
    if not os.path.exists(args.data_backend_config):
        raise FileNotFoundError(
            f"Data backend config file {args.data_backend_config} not found."
        )
    info_log(f"Loading data backend config from {args.data_backend_config}")
    with open(args.data_backend_config, "r", encoding="utf-8") as f:
        data_backend_config = json.load(f)
    if len(data_backend_config) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )

    text_embed_backends = {}
    image_embed_backends = {}

    ###                                            ###
    #    now we configure the text embed backends    #
    ###                                            ###
    default_text_embed_backend_id = None
    text_embed_cache_dir_paths = []
    for backend in data_backend_config:
        dataset_type = backend.get("dataset_type", None)
        if dataset_type is None or dataset_type != "text_embeds":
            # Skip configuration of image data backends. It is done later.
            continue
        if ("disabled" in backend and backend["disabled"]) or (
            "disable" in backend and backend["disable"]
        ):
            info_log(f"Skipping disabled data backend {backend['id']} in config file.")
            continue

        info_log(f'Configuring text embed backend: {backend["id"]}')
        if backend.get("default", None):
            if default_text_embed_backend_id is not None:
                raise ValueError(
                    "Only one text embed backend can be marked as default."
                )
            default_text_embed_backend_id = backend["id"]
        # Retrieve some config file overrides for commandline arguments,
        #  there currently isn't much for text embeds.
        init_backend = init_backend_config(backend, args, accelerator)
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])
        if backend["type"] == "local":
            text_embed_cache_dir_paths.append(
                backend.get("cache_dir", args.cache_dir_text)
            )
            init_backend["data_backend"] = get_local_backend(
                accelerator, init_backend["id"], compress_cache=args.compress_disk_cache
            )
            init_backend["cache_dir"] = backend["cache_dir"]
        elif backend["type"] == "aws":
            check_aws_config(backend)
            init_backend["data_backend"] = get_aws_backend(
                identifier=init_backend["id"],
                aws_bucket_name=backend["aws_bucket_name"],
                aws_region_name=backend["aws_region_name"],
                aws_endpoint_url=backend["aws_endpoint_url"],
                aws_access_key_id=backend["aws_access_key_id"],
                aws_secret_access_key=backend["aws_secret_access_key"],
                accelerator=accelerator,
                max_pool_connections=backend.get(
                    "max_pool_connections", args.aws_max_pool_connections
                ),
            )
            # S3 buckets use the aws_data_prefix as their prefix/ for all data.
            # Ensure we have a trailing slash on the prefix:
            init_backend["cache_dir"] = backend.get(
                "aws_data_prefix", backend.get("cache_dir", args.cache_dir_text)
            )
        elif backend["type"] == "csv":
            raise ValueError("Cannot use CSV backend for text embed storage.")
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")

        preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
        if not preserve_data_backend_cache and accelerator.is_local_main_process:
            StateTracker.delete_cache_files(
                data_backend_id=init_backend["id"],
                preserve_data_backend_cache=preserve_data_backend_cache,
            )

        # Generate a TextEmbeddingCache object
        init_backend["text_embed_cache"] = TextEmbeddingCache(
            id=init_backend["id"],
            data_backend=init_backend["data_backend"],
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            accelerator=accelerator,
            cache_dir=init_backend.get("cache_dir", args.cache_dir_text),
            model_type=StateTracker.get_model_family(),
            write_batch_size=backend.get("write_batch_size", args.write_batch_size),
        )
        init_backend["text_embed_cache"].set_webhook_handler(
            StateTracker.get_webhook_handler()
        )
        with accelerator.main_process_first():
            init_backend["text_embed_cache"].discover_all_files()
        accelerator.wait_for_everyone()

        if backend.get("default", False):
            # The default embed cache will be used for eg. validation prompts.
            StateTracker.set_default_text_embed_cache(init_backend["text_embed_cache"])
            logger.debug(f"Set the default text embed cache to {init_backend['id']}.")
            # We will compute the null embedding for caption dropout here.
            info_log("Pre-computing null embedding")
            with accelerator.main_process_first():
                init_backend["text_embed_cache"].compute_embeddings_for_prompts(
                    [""], return_concat=False, load_from_cache=False
                )
            time.sleep(5)
            accelerator.wait_for_everyone()
        if args.caption_dropout_probability == 0.0:
            logger.warning(
                "Not using caption dropout will potentially lead to overfitting on captions, eg. CFG will not work very well. Set --caption_dropout_probability=0.1 as a recommended value."
            )

        # We don't compute the text embeds at this time, because we do not really have any captions available yet.
        text_embed_backends[init_backend["id"]] = init_backend

    if not text_embed_backends:
        raise ValueError(
            "Your dataloader config must contain at least one image dataset AND at least one text_embed dataset."
            " See this link for more information about dataset_type: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
        )
    if not default_text_embed_backend_id and len(text_embed_backends) > 1:
        raise ValueError(
            f"You have {len(text_embed_backends)} text_embed dataset{'s' if len(text_embed_backends) > 1 else ''}, but no default text embed was defined."
            "\nPlease set default: true on one of the text_embed datasets, as this will be the location of global embeds (validation prompts, etc)."
            "\nSee this link for more information on how to configure a default text embed dataset: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
        )
    elif not default_text_embed_backend_id:
        logger.warning(
            f"No default text embed was defined, using {list(text_embed_backends.keys())[0]} as the default."
            " See this page for information about the default text embed backend: https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options"
        )
        default_text_embed_backend_id = list(text_embed_backends.keys())[0]
    info_log("Completed loading text embed services.")

    ###                                             ###
    #    now we configure the image embed backends    #
    ###                                             ###
    for backend in data_backend_config:
        dataset_type = backend.get("dataset_type", None)
        if dataset_type is None or dataset_type != "image_embeds":
            continue
        if ("disabled" in backend and backend["disabled"]) or (
            "disable" in backend and backend["disable"]
        ):
            info_log(f"Skipping disabled data backend {backend['id']} in config file.")
            continue

        info_log(f'Configuring VAE image embed backend: {backend["id"]}')
        # Retrieve some config file overrides for commandline arguments,
        #  there currently isn't much for text embeds.
        init_backend = init_backend_config(backend, args, accelerator)
        existing_config = StateTracker.get_data_backend_config(init_backend["id"])
        if existing_config is not None and existing_config != {}:
            raise ValueError(
                f"You can only have one backend named {init_backend['id']}"
            )
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])
        if backend["type"] == "local":
            init_backend["data_backend"] = get_local_backend(
                accelerator, init_backend["id"], compress_cache=args.compress_disk_cache
            )
        elif backend["type"] == "aws":
            check_aws_config(backend)
            init_backend["data_backend"] = get_aws_backend(
                identifier=init_backend["id"],
                aws_bucket_name=backend["aws_bucket_name"],
                aws_region_name=backend["aws_region_name"],
                aws_endpoint_url=backend["aws_endpoint_url"],
                aws_access_key_id=backend["aws_access_key_id"],
                aws_secret_access_key=backend["aws_secret_access_key"],
                accelerator=accelerator,
                max_pool_connections=backend.get(
                    "max_pool_connections", args.aws_max_pool_connections
                ),
            )
            # S3 buckets use the aws_data_prefix as their prefix/ for all data.
            # Ensure we have a trailing slash on the prefix:
            init_backend["cache_dir"] = backend.get("aws_data_prefix", None)
        elif backend["type"] == "csv":
            raise ValueError("Cannot use CSV backend for image embed storage.")
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")

        preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
        if not preserve_data_backend_cache and accelerator.is_local_main_process:
            StateTracker.delete_cache_files(
                data_backend_id=init_backend["id"],
                preserve_data_backend_cache=preserve_data_backend_cache,
            )

        image_embed_backends[init_backend["id"]] = init_backend

    ###                                       ###
    #    now we configure the image backends    #
    ###                                       ###
    vae_cache_dir_paths = []  # tracking for duplicates
    for backend in data_backend_config:
        dataset_type = backend.get("dataset_type", None)
        if dataset_type is not None and (
            dataset_type != "image" and dataset_type != "conditioning"
        ):
            # Skip configuration of text embed backends. It is done earlier.
            continue
        if ("disabled" in backend and backend["disabled"]) or (
            "disable" in backend and backend["disable"]
        ):
            info_log(f"Skipping disabled data backend {backend['id']} in config file.")
            continue
        # For each backend, we will create a dict to store all of its components in.
        if (
            "id" not in backend
            or backend["id"] == ""
            or backend["id"] in StateTracker.get_data_backends()
        ):
            raise ValueError("Each dataset needs a unique 'id' field.")
        info_log(f"Configuring data backend: {backend['id']}")
        conditioning_type = backend.get("conditioning_type")
        if (
            backend.get("dataset_type") == "conditioning"
            or conditioning_type is not None
        ):
            backend["dataset_type"] = "conditioning"
        resolution_type = backend.get("resolution_type", args.resolution_type)
        if resolution_type == "pixel_area":
            pixel_edge_length = backend.get("resolution", int(args.resolution))
            if pixel_edge_length is None or (
                type(pixel_edge_length) is not int
                or not str(pixel_edge_length).isdigit()
            ):
                raise ValueError(
                    f"Resolution type 'pixel_area' requires a 'resolution' field to be set in the backend config using an integer in the format: 1024, but {pixel_edge_length} was given"
                )
            # we'll convert pixel_area to area
            backend["resolution_type"] = "area"
            backend["resolution"] = (pixel_edge_length * pixel_edge_length) / (1000**2)
            # convert the other megapixel values.
            if (
                backend.get("maximum_image_size", None) is not None
                and backend["maximum_image_size"] > 0
            ):
                backend["maximum_image_size"] = (
                    backend["maximum_image_size"] * backend["maximum_image_size"]
                ) / 1_000_000
            if (
                backend.get("target_downsample_size", None) is not None
                and backend["target_downsample_size"] > 0
            ):
                backend["target_downsample_size"] = (
                    backend["target_downsample_size"]
                    * backend["target_downsample_size"]
                ) / 1_000_000
            if (
                backend.get("minimum_image_size", None) is not None
                and backend["minimum_image_size"] > 0
            ):
                backend["minimum_image_size"] = (
                    backend["minimum_image_size"] * backend["minimum_image_size"]
                ) / 1_000_000

        # Retrieve some config file overrides for commandline arguments, eg. cropping
        init_backend = init_backend_config(backend, args, accelerator)
        StateTracker.set_data_backend_config(
            data_backend_id=init_backend["id"],
            config=init_backend["config"],
        )

        preserve_data_backend_cache = backend.get("preserve_data_backend_cache", False)
        if not preserve_data_backend_cache:
            StateTracker.delete_cache_files(
                data_backend_id=init_backend["id"],
                preserve_data_backend_cache=preserve_data_backend_cache,
            )
        StateTracker.load_aspect_resolution_map(
            dataloader_resolution=init_backend["config"]["resolution"],
        )

        if backend["type"] == "local":
            init_backend["data_backend"] = get_local_backend(
                accelerator, init_backend["id"], compress_cache=args.compress_disk_cache
            )
            init_backend["instance_data_dir"] = backend.get(
                "instance_data_dir", backend.get("instance_data_root")
            )
            if init_backend["instance_data_dir"] is None:
                raise ValueError(
                    "A local backend requires instance_data_dir be defined and pointing to the image data directory."
                )
            # Remove trailing slash
            if (
                init_backend["instance_data_dir"] is not None
                and init_backend["instance_data_dir"][-1] == "/"
            ):
                init_backend["instance_data_dir"] = init_backend["instance_data_dir"][
                    :-1
                ]
        elif backend["type"] == "aws":
            check_aws_config(backend)
            init_backend["data_backend"] = get_aws_backend(
                identifier=init_backend["id"],
                aws_bucket_name=backend["aws_bucket_name"],
                aws_region_name=backend["aws_region_name"],
                aws_endpoint_url=backend["aws_endpoint_url"],
                aws_access_key_id=backend["aws_access_key_id"],
                aws_secret_access_key=backend["aws_secret_access_key"],
                accelerator=accelerator,
                compress_cache=args.compress_disk_cache,
                max_pool_connections=backend.get(
                    "max_pool_connections", args.aws_max_pool_connections
                ),
            )
            # S3 buckets use the aws_data_prefix as their prefix/ for all data.
            init_backend["instance_data_dir"] = backend.get("aws_data_prefix", "")
        elif backend["type"] == "csv":
            check_csv_config(backend=backend, args=args)
            init_backend["data_backend"] = get_csv_backend(
                accelerator=accelerator,
                id=backend["id"],
                csv_file=backend["csv_file"],
                csv_cache_dir=backend["csv_cache_dir"],
                compress_cache=args.compress_disk_cache,
                hash_filenames=backend.get("hash_filenames", False),
            )
            # init_backend["instance_data_dir"] = backend.get("instance_data_dir", backend.get("instance_data_root", backend.get("csv_cache_dir")))
            init_backend["instance_data_dir"] = None
            # if init_backend["instance_data_dir"] is None:
            #     raise ValueError("CSV backend requires one of instance_data_dir, instance_data_root or csv_cache_dir to be set, as we require a location to place metadata lists.")
            # Remove trailing slash
            if (
                init_backend["instance_data_dir"] is not None
                and init_backend["instance_data_dir"][-1] == "/"
            ):
                init_backend["instance_data_dir"] = init_backend["instance_data_dir"][
                    :-1
                ]
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")

        # Assign a TextEmbeddingCache to this dataset. it might be undefined.
        text_embed_id = backend.get(
            "text_embeds",
            backend.get("text_embed_cache", default_text_embed_backend_id),
        )
        if text_embed_id not in text_embed_backends:
            raise ValueError(
                f"Text embed backend {text_embed_id} not found in data backend config file."
            )
        # Do we have a specific VAE embed backend?
        image_embed_backend_id = backend.get("image_embeds", None)
        image_embed_data_backend = init_backend
        if image_embed_backend_id is not None:
            if image_embed_backend_id not in image_embed_backends:
                raise ValueError(
                    f"Could not find image embed backend ID in multidatabackend config: {image_embed_backend_id}"
                )
            image_embed_data_backend = image_embed_backends[image_embed_backend_id]
        info_log(f"(id={init_backend['id']}) Loading bucket manager.")
        metadata_backend_args = {}
        metadata_backend = backend.get("metadata_backend", "discovery")
        if metadata_backend == "json" or metadata_backend == "discovery":
            from videotuna.third_party.flux.metadata.backends.discovery import (
                DiscoveryMetadataBackend,
            )

            BucketManager_cls = DiscoveryMetadataBackend
        elif metadata_backend == "parquet":
            from videotuna.third_party.flux.metadata.backends.parquet import (
                ParquetMetadataBackend,
            )

            BucketManager_cls = ParquetMetadataBackend
            metadata_backend_args["parquet_config"] = backend.get("parquet", None)
            if not metadata_backend_args["parquet_config"]:
                raise ValueError(
                    "Parquet metadata backend requires a 'parquet' field in the backend config containing required fields for configuration."
                )
        else:
            raise ValueError(f"Unknown metadata backend type: {metadata_backend}")

        init_backend["metadata_backend"] = BucketManager_cls(
            id=init_backend["id"],
            instance_data_dir=init_backend["instance_data_dir"],
            data_backend=init_backend["data_backend"],
            accelerator=accelerator,
            resolution=backend.get("resolution", args.resolution),
            minimum_image_size=backend.get(
                "minimum_image_size", args.minimum_image_size
            ),
            resolution_type=backend.get("resolution_type", args.resolution_type),
            batch_size=args.train_batch_size,
            metadata_update_interval=backend.get(
                "metadata_update_interval", args.metadata_update_interval
            ),
            cache_file=os.path.join(
                backend.get(
                    "instance_data_dir",
                    backend.get("csv_cache_dir", backend.get("aws_data_prefix", "")),
                ),
                "aspect_ratio_bucket_indices",
            ),
            metadata_file=os.path.join(
                backend.get(
                    "instance_data_dir",
                    backend.get("csv_cache_dir", backend.get("aws_data_prefix", "")),
                ),
                "aspect_ratio_bucket_metadata",
            ),
            delete_problematic_images=args.delete_problematic_images or False,
            delete_unwanted_images=backend.get(
                "delete_unwanted_images", args.delete_unwanted_images
            ),
            cache_file_suffix=backend.get("cache_file_suffix", init_backend["id"]),
            repeats=init_backend["config"].get("repeats", 0),
            **metadata_backend_args,
        )

        if (
            "aspect" not in args.skip_file_discovery
            and "aspect" not in backend.get("skip_file_discovery", "")
            and conditioning_type not in ["mask", "controlnet"]
        ):
            if accelerator.is_local_main_process:
                info_log(
                    f"(id={init_backend['id']}) Refreshing aspect buckets on main process."
                )
                init_backend["metadata_backend"].refresh_buckets(rank_info())
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            info_log(
                f"(id={init_backend['id']}) Reloading bucket manager cache on subprocesses."
            )
            init_backend["metadata_backend"].reload_cache()
        accelerator.wait_for_everyone()
        if init_backend["metadata_backend"].has_single_underfilled_bucket():
            raise Exception(
                f"Cannot train using a dataset that has a single bucket with fewer than {args.train_batch_size} images."
                f" You have to reduce your batch size, or increase your dataset size (id={init_backend['id']})."
            )
        # Now split the contents of these buckets between all processes
        init_backend["metadata_backend"].split_buckets_between_processes(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        # Check if there is an existing 'config' in the metadata_backend.config
        excluded_keys = [
            "probability",
            "repeats",
            "ignore_epochs",
            "caption_filter_list",
            "vae_cache_clear_each_epoch",
            "caption_strategy",
            "maximum_image_size",
            "target_downsample_size",
            "parquet",
        ]
        # we will set the latest version by default.
        current_config_version = latest_config_version()
        if init_backend["metadata_backend"].config != {}:
            prev_config = init_backend["metadata_backend"].config
            # if the prev config used an old default config version, we will update defaults here.
            current_config_version = prev_config.get("config_version", None)
            if current_config_version is None:
                # backwards compatibility for non-versioned config files, so that we do not enable life-changing options.
                current_config_version = 1

            logger.debug(
                f"Found existing config (version={current_config_version}): {prev_config}"
            )
            logger.debug(f"Comparing against new config: {init_backend['config']}")
            # Check if any values differ between the 'backend' values and the 'config' values:
            for key, _ in prev_config.items():
                logger.debug(f"Checking config key: {key}")
                if key not in excluded_keys:
                    if key in backend and prev_config[key] != backend[key]:
                        # if not args.override_dataset_config:
                        #     raise Exception(
                        #         f"Dataset {init_backend['id']} has inconsistent config, and --override_dataset_config was not provided."
                        #         f"\n-> Expected value {key}={prev_config.get(key)} differs from current value={backend.get(key)}."
                        #         f"\n-> Recommended action is to correct the current config values to match the values that were used to create this dataset:"
                        #         f"\n{prev_config}"
                        #     )
                        # else:
                        #    logger.warning(
                        #        f"Overriding config value {key}={prev_config[key]} with {backend[key]}"
                        #    )
                        #    prev_config[key] = backend[key]
                        logger.warning(
                            f"Overriding config value {key}={prev_config[key]} with {backend[key]}"
                        )
                        prev_config[key] = backend[key]
                    elif key not in backend:
                        if should_log():
                            logger.warning(
                                f"Key {key} not found in the current backend config, using the existing value '{prev_config[key]}'."
                            )
                        init_backend["config"][key] = prev_config[key]

        init_backend["config"]["config_version"] = current_config_version
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])
        info_log(f"Configured backend: {init_backend}")

        print_bucket_info(init_backend["metadata_backend"])
        if len(init_backend["metadata_backend"]) == 0 and conditioning_type is None:
            raise Exception(
                f"No images were discovered by the bucket manager in the dataset: {init_backend['id']}."
            )

        use_captions = True
        is_regularisation_data = backend.get(
            "is_regularisation_data", backend.get("is_regularization_data", False)
        )
        if "only_instance_prompt" in backend and backend["only_instance_prompt"]:
            use_captions = False
        elif args.only_instance_prompt:
            use_captions = False
        init_backend["train_dataset"] = MultiAspectDataset(
            id=init_backend["id"],
            datasets=[init_backend["metadata_backend"]],
            is_regularisation_data=is_regularisation_data,
        )

        if "deepfloyd" in args.model_type:
            if init_backend["metadata_backend"].resolution_type == "area":
                logger.warning(
                    "Resolution type is 'area', but should be 'pixel' for DeepFloyd. Unexpected results may occur."
                )
                if init_backend["metadata_backend"].resolution > 0.25:
                    logger.warning(
                        "Resolution is greater than 0.25 megapixels. This may lead to unconstrained memory requirements."
                    )
            if init_backend["metadata_backend"].resolution_type == "pixel":
                if (
                    "stage2" not in args.model_type
                    and init_backend["metadata_backend"].resolution > 64
                ):
                    logger.warning(
                        "Resolution is greater than 64 pixels, which will possibly lead to poor quality results."
                    )

        if "deepfloyd-stage2" in args.model_type:
            # Resolution must be at least 256 for Stage II.
            if init_backend["metadata_backend"].resolution < 256:
                logger.warning(
                    "Increasing resolution to 256, as is required for DF Stage II."
                )

        init_backend["sampler"] = MultiAspectSampler(
            id=init_backend["id"],
            metadata_backend=init_backend["metadata_backend"],
            data_backend=init_backend["data_backend"],
            accelerator=accelerator,
            batch_size=args.train_batch_size,
            debug_aspect_buckets=args.debug_aspect_buckets,
            delete_unwanted_images=backend.get(
                "delete_unwanted_images", args.delete_unwanted_images
            ),
            resolution=backend.get("resolution", args.resolution),
            resolution_type=backend.get("resolution_type", args.resolution_type),
            caption_strategy=backend.get("caption_strategy", args.caption_strategy),
            use_captions=use_captions,
            prepend_instance_prompt=backend.get(
                "prepend_instance_prompt", args.prepend_instance_prompt
            ),
            instance_prompt=backend.get("instance_prompt", args.instance_prompt),
            conditioning_type=conditioning_type,
            is_regularisation_data=is_regularisation_data,
        )
        if init_backend["sampler"].caption_strategy == "parquet":
            configure_parquet_database(backend, args, init_backend["data_backend"])
        init_backend["train_dataloader"] = torch.utils.data.DataLoader(
            init_backend["train_dataset"],
            batch_size=1,  # The sampler handles batching
            shuffle=False,  # The sampler handles shuffling
            sampler=init_backend["sampler"],
            collate_fn=lambda examples: collate_fn(examples),
            num_workers=0,
            persistent_workers=False,
        )

        init_backend["text_embed_cache"] = text_embed_backends[text_embed_id][
            "text_embed_cache"
        ]
        prepend_instance_prompt = backend.get(
            "prepend_instance_prompt", args.prepend_instance_prompt
        )
        instance_prompt = backend.get("instance_prompt", args.instance_prompt)
        if prepend_instance_prompt and instance_prompt is None:
            raise ValueError(
                f"Backend {init_backend['id']} has prepend_instance_prompt=True, but no instance_prompt was provided. You must provide an instance_prompt, or disable this option."
            )

        # Update the backend registration here so the metadata backend can be found.
        StateTracker.register_data_backend(init_backend)

        # We get captions from the IMAGE dataset. Not the text embeds dataset.
        if (
            conditioning_type != "mask"
            and "text" not in args.skip_file_discovery
            and "text" not in backend.get("skip_file_discovery", "")
        ):
            info_log(f"(id={init_backend['id']}) Collecting captions.")
            captions = PromptHandler.get_all_captions(
                data_backend=init_backend["data_backend"],
                instance_data_dir=init_backend["instance_data_dir"],
                prepend_instance_prompt=prepend_instance_prompt,
                instance_prompt=instance_prompt,
                use_captions=use_captions,
                caption_strategy=backend.get("caption_strategy", args.caption_strategy),
            )
            logger.debug(
                f"Pre-computing text embeds / updating cache. We have {len(captions)} captions to process, though these will be filtered next."
            )
            caption_strategy = backend.get("caption_strategy", args.caption_strategy)
            info_log(
                f"(id={init_backend['id']}) Initialise text embed pre-computation using the {caption_strategy} caption strategy. We have {len(captions)} captions to process."
            )
            init_backend["text_embed_cache"].compute_embeddings_for_prompts(
                captions, return_concat=False, load_from_cache=False
            )
            info_log(
                f"(id={init_backend['id']}) Completed processing {len(captions)} captions."
            )

        # Register the backend here so the sampler can be found.
        StateTracker.register_data_backend(init_backend)

        default_hash_option = True
        hash_filenames = init_backend["config"].get(
            "hash_filenames", default_hash_option
        )
        init_backend["config"]["hash_filenames"] = hash_filenames
        StateTracker.set_data_backend_config(init_backend["id"], init_backend["config"])
        logger.debug(f"Hashing filenames: {hash_filenames}")

        if (
            "deepfloyd" not in StateTracker.get_args().model_type
            and conditioning_type not in ["mask", "controlnet"]
        ):
            info_log(f"(id={init_backend['id']}) Creating VAE latent cache.")
            vae_cache_dir = backend.get("cache_dir_vae", None)
            if vae_cache_dir in vae_cache_dir_paths:
                raise ValueError(
                    f"VAE image embed cache directory {backend.get('cache_dir_vae')} is the same as another VAE image embed cache directory. This is not allowed, the trainer will get confused and sleepy and wake up in a distant place with no memory and no money for a taxi ride back home, forever looking in the mirror and wondering who they are. This should be avoided."
                )
            vae_cache_dir_paths.append(vae_cache_dir)

            if (
                vae_cache_dir is not None
                and vae_cache_dir in text_embed_cache_dir_paths
            ):
                raise ValueError(
                    f"VAE image embed cache directory {backend.get('cache_dir_vae')} is the same as the text embed cache directory. This is not allowed, the trainer will get confused."
                )
            init_backend["vaecache"] = VAECache(
                id=init_backend["id"],
                vae=StateTracker.get_vae(),
                accelerator=accelerator,
                metadata_backend=init_backend["metadata_backend"],
                image_data_backend=init_backend["data_backend"],
                cache_data_backend=image_embed_data_backend["data_backend"],
                instance_data_dir=init_backend["instance_data_dir"],
                delete_problematic_images=backend.get(
                    "delete_problematic_images", args.delete_problematic_images
                ),
                resolution=backend.get("resolution", args.resolution),
                resolution_type=backend.get("resolution_type", args.resolution_type),
                maximum_image_size=backend.get(
                    "maximum_image_size",
                    args.maximum_image_size
                    or backend.get("resolution", args.resolution) * 1.5,
                ),
                target_downsample_size=backend.get(
                    "target_downsample_size",
                    args.target_downsample_size
                    or backend.get("resolution", args.resolution) * 1.25,
                ),
                minimum_image_size=backend.get(
                    "minimum_image_size",
                    args.minimum_image_size,
                ),
                vae_batch_size=backend.get("vae_batch_size", args.vae_batch_size),
                write_batch_size=backend.get("write_batch_size", args.write_batch_size),
                read_batch_size=backend.get("read_batch_size", args.read_batch_size),
                cache_dir=backend.get("cache_dir_vae", args.cache_dir_vae),
                max_workers=backend.get("max_workers", args.max_workers),
                process_queue_size=backend.get(
                    "image_processing_batch_size", args.image_processing_batch_size
                ),
                vae_cache_ondemand=args.vae_cache_ondemand,
                hash_filenames=hash_filenames,
            )
            init_backend["vaecache"].set_webhook_handler(
                StateTracker.get_webhook_handler()
            )

            if not args.vae_cache_ondemand:
                info_log(f"(id={init_backend['id']}) Discovering cache objects..")
                if accelerator.is_local_main_process:
                    init_backend["vaecache"].discover_all_files()
                accelerator.wait_for_everyone()
            all_image_files = StateTracker.get_image_files(
                data_backend_id=init_backend["id"]
            )
            init_backend["vaecache"].build_vae_cache_filename_map(
                all_image_files=all_image_files
            )

        if (
            (
                "metadata" not in args.skip_file_discovery
                or "metadata" not in backend.get("skip_file_discovery", "")
            )
            and accelerator.is_main_process
            and backend.get("scan_for_errors", False)
            and "deepfloyd" not in StateTracker.get_args().model_type
            and conditioning_type not in ["mask", "controlnet"]
        ):
            info_log(
                f"Beginning error scan for dataset {init_backend['id']}. Set 'scan_for_errors' to False in the dataset config to disable this."
            )
            init_backend["metadata_backend"].handle_vae_cache_inconsistencies(
                vae_cache=init_backend["vaecache"],
                vae_cache_behavior=backend.get(
                    "vae_cache_scan_behaviour", args.vae_cache_scan_behaviour
                ),
            )
            init_backend["metadata_backend"].scan_for_metadata()

        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            init_backend["metadata_backend"].load_image_metadata()
        accelerator.wait_for_everyone()

        if (
            not args.vae_cache_ondemand
            and "vaecache" in init_backend
            and "vae" not in args.skip_file_discovery
            and "vae" not in backend.get("skip_file_discovery", "")
            and "deepfloyd" not in StateTracker.get_args().model_type
            and conditioning_type not in ["mask", "controlnet"]
        ):
            init_backend["vaecache"].discover_unprocessed_files()
            if not args.vae_cache_ondemand:
                init_backend["vaecache"].process_buckets()
            logger.debug(f"Encoding images during training: {args.vae_cache_ondemand}")
            accelerator.wait_for_everyone()

        info_log(f"Configured backend: {init_backend}")

        StateTracker.register_data_backend(init_backend)
        init_backend["metadata_backend"].save_cache()

    # For each image backend, connect it to its conditioning backend.
    for backend in data_backend_config:
        dataset_type = backend.get("dataset_type", "image")
        if dataset_type is not None and dataset_type != "image":
            # Skip configuration of conditioning/text data backends. It is done earlier.
            continue
        if ("disabled" in backend and backend["disabled"]) or (
            "disable" in backend and backend["disable"]
        ):
            info_log(f"Skipping disabled data backend {backend['id']} in config file.")
            continue
        if "conditioning_data" in backend and backend[
            "conditioning_data"
        ] not in StateTracker.get_data_backends(_type="conditioning"):
            raise ValueError(
                f"Conditioning data backend {backend['conditioning_data']} not found in data backend list: {StateTracker.get_data_backends()}."
            )
        if "conditioning_data" in backend:
            StateTracker.set_conditioning_dataset(
                backend["id"], backend["conditioning_data"]
            )
            info_log(
                f"Successfully configured conditioning image dataset for {backend['id']}"
            )

    if len(StateTracker.get_data_backends()) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )
    return StateTracker.get_data_backends()


def get_local_backend(
    accelerator, identifier: str, compress_cache: bool = False
) -> LocalDataBackend:
    """
    Get a local disk backend.

    Args:
        accelerator (Accelerator): A Huggingface Accelerate object.
        identifier (str): An identifier that links this data backend to its other components.
    Returns:
        LocalDataBackend: A LocalDataBackend object.
    """
    return LocalDataBackend(
        accelerator=accelerator, id=identifier, compress_cache=compress_cache
    )


def get_csv_backend(
    accelerator,
    id: str,
    csv_file: str,
    csv_cache_dir: str,
    url_column: str,
    caption_column: str,
    compress_cache: bool = False,
    hash_filenames: bool = False,
    shorten_filenames: bool = False,
) -> CSVDataBackend:
    from pathlib import Path

    return CSVDataBackend(
        accelerator=accelerator,
        id=id,
        csv_file=Path(csv_file),
        image_cache_loc=csv_cache_dir,
        url_column=url_column,
        caption_column=caption_column,
        compress_cache=compress_cache,
        shorten_filenames=shorten_filenames,
        hash_filenames=hash_filenames,
    )


def check_csv_config(backend: dict, args) -> None:
    required_keys = {
        "csv_file": "This is the path to the CSV file containing your image URLs.",
        "csv_cache_dir": "This is the path to your temporary cache files where images will be stored. This can grow quite large.",
        "csv_caption_column": "This is the column in your csv which contains the caption(s) for the samples.",
        "csv_url_column": "This is the column in your csv that contains image urls or paths.",
    }
    for key in required_keys.keys():
        if key not in backend:
            raise ValueError(
                f"Missing required key {key} in CSV backend config: {required_keys[key]}"
            )
    if not args.compress_disk_cache:
        logger.warning(
            "You can save more disk space for cache objects by providing --compress_disk_cache and recreating its contents"
        )
    caption_strategy = backend.get("caption_strategy")
    if caption_strategy is None or caption_strategy != "csv":
        raise ValueError("CSV backend requires a caption_strategy of 'csv'.")


def check_aws_config(backend: dict) -> None:
    """
    Check the configuration for an AWS backend.

    Args:
        backend (dict): A dictionary of the backend configuration.
    Returns:
        None
    """
    required_keys = [
        "aws_bucket_name",
        "aws_region_name",
        "aws_endpoint_url",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]
    for key in required_keys:
        if key not in backend:
            raise ValueError(f"Missing required key {key} in AWS backend config.")


def get_aws_backend(
    aws_bucket_name: str,
    aws_region_name: str,
    aws_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    accelerator,
    identifier: str,
    compress_cache: bool = False,
    max_pool_connections: int = 128,
) -> S3DataBackend:
    return S3DataBackend(
        id=identifier,
        bucket_name=aws_bucket_name,
        accelerator=accelerator,
        region_name=aws_region_name,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        compress_cache=compress_cache,
        max_pool_connections=max_pool_connections,
    )


def select_dataloader_index(step, backends):
    # Generate weights for each backend based on some criteria
    weights = []
    backend_ids = []
    for backend_id, backend in backends.items():
        weight = get_backend_weight(backend_id, backend, step)
        weights.append(weight)
        backend_ids.append(backend_id)

    # Convert to a torch tensor for easy sampling
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()  # Normalize the weights
    if weights.sum() == 0:
        return None

    # Sample a backend index based on the weights
    chosen_index = torch.multinomial(weights, 1).item()
    chosen_backend_id = backend_ids[chosen_index]

    return chosen_backend_id


def get_backend_weight(backend_id, backend, step):
    backend_config = StateTracker.get_data_backend_config(backend_id)
    prob = backend_config.get("probability", 1)

    if StateTracker.get_args().data_backend_sampling == "uniform":
        return prob
    elif StateTracker.get_args().data_backend_sampling == "auto-weighting":
        # Get the dataset length (assuming you have a method or property to retrieve it)
        dataset_length = StateTracker.get_dataset_size(backend_id)

        # Calculate the weight based on dataset length
        length_factor = dataset_length / sum(
            StateTracker.get_dataset_size(b) for b in StateTracker.get_data_backends()
        )

        # Adjust the probability by length factor
        adjusted_prob = prob * length_factor

        disable_step = backend_config.get("disable_after_epoch_step", None)
        if disable_step:
            disable_step = int(disable_step)
        else:
            disable_step = float("inf")
        adjusted_prob = (
            0
            if int(step) > disable_step
            else max(0, adjusted_prob * (1 - step / disable_step))
        )

        return adjusted_prob
    else:
        raise ValueError(
            f"Unknown sampling weighting method: {StateTracker.get_args().data_backend_sampling}"
        )


def random_dataloader_iterator(step, backends: dict):
    prefetch_log_debug("Random dataloader iterator launched.")
    gradient_accumulation_steps = StateTracker.get_args().gradient_accumulation_steps
    logger.debug(f"Backends to select from {backends}")
    if backends == {}:
        logger.debug(
            "All dataloaders exhausted. Moving to next epoch in main training loop."
        )
        StateTracker.clear_exhausted_buckets()
        StateTracker.set_repeats(repeats=0)
        return False
    while backends:
        epoch_step = int(step / gradient_accumulation_steps)
        StateTracker.set_epoch_step(epoch_step)

        chosen_backend_id = select_dataloader_index(step, backends)
        if chosen_backend_id is None:
            logger.debug("No dataloader iterators were available.")
            break

        chosen_iter = iter(backends[chosen_backend_id])

        try:
            return next(chosen_iter)
        except MultiDatasetExhausted:
            # We may want to repeat the same dataset multiple times in a single epoch.
            # If so, we can just reset the iterator and keep going.
            repeats = StateTracker.get_data_backend_config(chosen_backend_id).get(
                "repeats", False
            )
            if (
                repeats
                and repeats > 0
                and StateTracker.get_repeats(chosen_backend_id) < repeats
            ):
                StateTracker.increment_repeats(chosen_backend_id)
                logger.debug(
                    f"Dataset (name={chosen_backend_id}) is now sampling its {StateTracker.get_repeats(chosen_backend_id)} repeat out of {repeats} total allowed."
                )
                continue
            logger.debug(
                f"Dataset (name={chosen_backend_id}) is now exhausted after {StateTracker.get_repeats(chosen_backend_id)} repeat(s). Removing from list."
            )
            del backends[chosen_backend_id]
            StateTracker.backend_exhausted(chosen_backend_id)
            StateTracker.set_repeats(data_backend_id=chosen_backend_id, repeats=0)
        finally:
            if not backends:
                logger.debug(
                    "All dataloaders exhausted. Moving to next epoch in main training loop."
                )
                StateTracker.clear_exhausted_buckets()
                return False


class BatchFetcher:
    def __init__(self, step, max_size=10, datasets={}):
        self.queue = queue.Queue(max_size)
        self.datasets = datasets
        self.keep_running = True
        self.step = step

    def start_fetching(self):
        thread = threading.Thread(target=self.fetch_responses)
        thread.start()
        return thread

    def fetch_responses(self):
        prefetch_log_debug("Launching retrieval thread.")
        while self.keep_running:
            if self.queue.qsize() < self.queue.maxsize:
                prefetch_log_debug(
                    f"Queue size: {self.queue.qsize()}. Fetching more data."
                )
                self.queue.put(random_dataloader_iterator(self.step, self.datasets))
                if self.queue.qsize() >= self.queue.maxsize:
                    prefetch_log_debug("Completed fetching data. Queue is full.")
                    continue
            else:
                time.sleep(0.5)
        prefetch_log_debug("Exiting retrieval thread.")

    def next_response(self, step: int):
        self.step = step
        if self.queue.empty():
            prefetch_log_debug("Queue is empty. Waiting for data.")
        while self.queue.empty():
            continue
        prefetch_log_debug("Queue has data. Yielding next item.")
        return self.queue.get()

    def stop_fetching(self):
        self.keep_running = False
