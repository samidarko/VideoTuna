import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image
import imageio

import torch
import numpy as np
from einops import rearrange, repeat
import torchvision.transforms as TT
import omegaconf
from omegaconf import OmegaConf





from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from sat.arguments import add_training_args, add_evaluation_args, add_data_args
from sat.arguments import set_random_seed



import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/cogvideo_sat"))
# from cogvideo_sat import diffusion_video
from diffusion_video import SATVideoDiffusionEngine

from arguments import get_args

import deepspeed
from sat.helpers import print_rank0


def add_sampling_config_args(parser):
    """Sampling configurations"""
    group = parser.add_argument_group("sampling", "Sampling Configurations")
    group.add_argument("--input-dir", type=str, default=None)
    group.add_argument("--sampling-image-size", type=list, default=[768, 1360])
    group.add_argument("--final-size", type=int, default=2048)
    group.add_argument("--sdedit", action="store_true")
    group.add_argument("--grid-num-rows", type=int, default=1)
    group.add_argument("--force-inference", action="store_true")
    group.add_argument("--lcm_steps", type=int, default=None)
    group.add_argument("--sampling-num-frames", type=int, default=22)
    group.add_argument("--sampling-fps", type=int, default=16)
    group.add_argument("--only-save-latents", type=bool, default=False)
    group.add_argument("--only-log-video-latents", type=bool, default=False)
    group.add_argument("--latent-channels", type=int, default=16)
    group.add_argument("--image2video", action="store_true")
    group.add_argument("--modeForScript", type=str, default="inference")
    group.add_argument("--batch_size", type=int, default=1)

    return parser

def add_model_config_args(parser):
    """Model arguments"""
    group = parser.add_argument_group("model", "model configuration")
    # group.add_argument("--base", type=str, nargs="*", help="config for input and saving", default="configs/005_cogvideox1.5/cogvideox1.5_5b_t2v.yaml")
    group.add_argument(
        "--model-parallel-size", type=int, default=1, help="size of the model parallel. only use if you are an expert."
    )
    group.add_argument("--force-pretrain", action="store_true", default=True)
    group.add_argument("--device", type=int, default=-1)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--log-image", type=bool, default=True)

    return parser

def initialize_distributed(args):
    """Initialize torch.distributed."""
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True
    # the automatic assignment of devices has been moved to arguments.py
    if args.device == "cpu":
        pass
    else:
        torch.cuda.set_device(args.device)
    # Call the init process
    init_method = "tcp://"
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    if args.world_size == 1:
        from sat.helpers import get_free_port

        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"
    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    init_method += args.master_ip + ":" + args.master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
    )

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Set vae context parallel group equal to model parallel group
    from sgm.util import set_context_parallel_group, initialize_context_parallel

    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)
    # mpu.initialize_model_parallel(1)
    # Optional DeepSpeed Activation Checkpointing Features
    if args.deepspeed:
        import deepspeed

        deepspeed.init_distributed(
            dist_backend=args.distributed_backend, world_size=args.world_size, rank=args.rank, init_method=init_method
        )
        # # It seems that it has no negative influence to configure it even without using checkpointing.
        # deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    else:
        # in model-only mode, we don't want to init deepspeed, but we still need to init the rng tracker for model_parallel, just because we save the seed by default when dropout.
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )

            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # default seed 1
        except Exception as e:
            from sat.helpers import print_rank0

            print_rank0(str(e), level="DEBUG")

    return True

def process_config_to_args(args):
    """Fetch args from only --base"""
    project_dir = os.path.join(os.path.dirname(__file__), "../")

    def extract_clean_path(base):
        base = base[0].strip('["]')
        clean_path = base.strip("[]").strip("'")
        return clean_path
    clean_path = extract_clean_path(args.base)

    args.base = [os.path.join(project_dir, clean_path)]


    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)

    args_config = config.pop("args", OmegaConf.create())
    for key in args_config:
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            arg = OmegaConf.to_object(args_config[key])
        else:
            arg = args_config[key]
        if hasattr(args, key):
            setattr(args, key, arg)

    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())
        args.model_config = model_config
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())
        args.data_config = data_config

    return args

def getArgs():
    parser = argparse.ArgumentParser(description="sat") 

    parser.add_argument("--load_transformer", type=str, default="checkpoints/cogvideo/CogVideoX1.5-5B-SAT/transformer_t2v")
    parser.add_argument("--input_type", type=str, default="txt")
    parser.add_argument("--input_file", type=str, default="configs/005_cogvideox1.5/prompt.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--base", type=str, nargs="*", help="config for input and saving", default="configs/005_cogvideox1.5/cogvideox1.5_5b_t2v.yaml")
    parser.add_argument("--mode_type", type=str, default="t2v")
    parser.add_argument("--sampling_num_frames", type=int, default=22)
    parser.add_argument("--image_folder", type=str, default="inputs/i2v/576x1024")



    parser = add_model_config_args(parser)
    parser = add_sampling_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    args_list = ['--base', parser.parse_args().base]
    args = parser.parse_args(args_list)
    args = process_config_to_args(args)

    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun

    if args.device == -1:
        if torch.cuda.device_count() == 0:
            args.device = "cpu"
        elif args.local_rank is not None:
            args.device = args.local_rank
        else:
            args.device = args.rank % torch.cuda.device_count()

    if args.local_rank != args.device and args.mode != "inference":
        raise ValueError(
            "LOCAL_RANK (default 0) and args.device inconsistent. "
            "This can only happens in inference mode. "
            "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
        )

    if args.rank == 0:
        print_rank0("using world size: {}".format(args.world_size))


    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        if override_deepspeed_config:  # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config

    initialize_distributed(args)
    args.seed = args.seed + mpu.get_data_parallel_rank()
    set_random_seed(args.seed)

    args.load = parser.parse_args().load_transformer
    args.input_type = parser.parse_args().input_type
    args.input_file = parser.parse_args().input_file
    args.output_dir = parser.parse_args().output_dir
    args.batch_size = 1
    args.bf16 = True

    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.force_inference = True
    args.mode = "inference"
    args.sampling_num_frames = parser.parse_args().sampling_num_frames

    if parser.parse_args().mode_type == "t2v":
        args.image2video = False
        args.sampling_image_size = [768, 1360]
    else:
        args.image2video = True
        args.model_config.network_config.params.in_channels = 32
        args.image_path = parser.parse_args().image_folder
        
    return args



def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
        elif key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(math.prod(N))
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(device, dtype=torch.half)
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    sample_func = model.sample
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    T, C = args.sampling_num_frames, args.latent_channels
    with torch.no_grad():
        def get_images_in_list(folder_path, extensions=("jpg", "png")):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
            files = sorted(files)
            image_list = [os.path.join(folder_path, file) for file in files]
            return image_list
        
        images = get_images_in_list(args.image_folder)
        counter = 0
        for text, cnt in tqdm(data_iter):
            def get_images_in_list(folder_path, extensions=("jpg", "png")):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
                files = sorted(files)
                image_list = [os.path.join(folder_path, file) for file in files]
                return image_list
                
            if args.image2video:
                # use with input image shape
                # text, image_path = text.split("@@")
                # assert os.path.exists(image_path), image_path
                # image = Image.open(image_path).convert("RGB")
                image_path = images[counter]
                counter += 1
                
                assert os.path.exists(image_path), image_path
                image = Image.open(image_path).convert("RGB")
                (img_W, img_H) = image.size

                def nearest_multiple_of_16(n):
                    lower_multiple = (n // 16) * 16
                    upper_multiple = (n // 16 + 1) * 16
                    if abs(n - lower_multiple) < abs(n - upper_multiple):
                        return lower_multiple
                    else:
                        return upper_multiple

                if img_H < img_W:
                    H = 96
                    W = int(nearest_multiple_of_16(img_W / img_H * H * 8)) // 8
                else:
                    W = 96
                    H = int(nearest_multiple_of_16(img_H / img_W * W * 8)) // 8
                chained_trainsforms = []
                chained_trainsforms.append(TT.Resize(size=[int(H * 8), int(W * 8)], interpolation=1))
                chained_trainsforms.append(TT.ToTensor())
                transform = TT.Compose(chained_trainsforms)
                image = transform(image).unsqueeze(0).to("cuda")
                image = image * 2.0 - 1.0
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image / model.scale_factor
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H, W)
                image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
            else:
                image_size = args.sampling_image_size
                H, W = image_size[0], image_size[1]
                F = 8  # 8x downsampled
                image = None

            text_cast = [text]
            mp_size = mpu.get_model_parallel_world_size()
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast_object_list(text_cast, src=src, group=mpu.get_model_parallel_group())
            text = text_cast[0]
            value_dict = {"prompt": text, "negative_prompt": "", "num_frames": torch.tensor(T).unsqueeze(0)}

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            if args.image2video:
                c["concat"] = image
                uc["concat"] = image

            for index in range(args.batch_size):
                if args.image2video:
                    samples_z = sample_func(
                        c, uc=uc, batch_size=1, shape=(T, C, H, W), ofs=torch.tensor([2.0]).to("cuda")
                    )
                else:
                    samples_z = sample_func(
                        c,
                        uc=uc,
                        batch_size=1,
                        shape=(T, C, H // F, W // F),
                    ).to("cuda")

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                if args.only_save_latents:
                    samples_z = 1.0 / model.scale_factor * samples_z
                    save_path = os.path.join(
                        args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:120], str(index)
                    )
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(samples_z, os.path.join(save_path, "latent.pt"))
                    with open(os.path.join(save_path, "text.txt"), "w") as f:
                        f.write(text)
                else:
                    samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                    samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                    save_path = os.path.join(
                        args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:120], str(index)
                    )
                    if mpu.get_model_parallel_rank() == 0:
                        save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)


if __name__ == "__main__":
    args = getArgs()
    sampling_main(args, model_cls=SATVideoDiffusionEngine)





