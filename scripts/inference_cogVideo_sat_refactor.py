import argparse
import math
import os
import sys
from typing import List, Union

import imageio
import numpy as np
import omegaconf
import torch
import torchvision.transforms as TT
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from sat import mpu
from sat.arguments import (
    add_data_args,
    add_evaluation_args,
    add_training_args,
    set_random_seed,
)
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../videotuna/cogvideo_sat"))
import datetime

from arguments import getArgs

# from cogvideo_sat import diffusion_video
from diffusion_video import SATVideoDiffusionEngine

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None
):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"prompt-{key:04d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def main(args, model_cls):
    model = get_model(args, model_cls) if isinstance(model_cls, type) else model_cls
    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "txt":
        rank, world_size = (
            mpu.get_data_parallel_rank(),
            mpu.get_data_parallel_world_size(),
        )
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError("Only 'txt' input_type is supported.")

    sample_func = model.sample
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    T, C = args.sampling_num_frames, args.latent_channels
    counter = 0

    def get_images_in_list(folder_path, extensions=("jpg", "png")):
        files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(extensions)
        )
        return [os.path.join(folder_path, file) for file in files]

    def nearest_multiple_of_16(n):
        return int(min(((n // 16) * 16, (n // 16 + 1) * 16), key=lambda x: abs(n - x)))

    images = get_images_in_list(args.image_folder) if args.image2video else None

    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            if args.image2video:
                image_path = images[counter]
                counter += 1
                assert os.path.exists(
                    image_path
                ), f"Image path does not exist: {image_path}"

                image = Image.open(image_path).convert("RGB")
                img_W, img_H = image.size
                H, W = (
                    (96, nearest_multiple_of_16(img_W / img_H * 96 * 8) // 8)
                    if img_H < img_W
                    else (nearest_multiple_of_16(img_H / img_W * 96 * 8) // 8, 96)
                )

                transform = TT.Compose(
                    [
                        TT.Resize(size=[int(H * 8), int(W * 8)], interpolation=1),
                        TT.ToTensor(),
                    ]
                )
                image = transform(image).unsqueeze(0).to("cuda") * 2.0 - 1.0
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None) / model.scale_factor
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H, W)
                image = torch.cat(
                    [
                        image,
                        torch.zeros(pad_shape, device=image.device, dtype=image.dtype),
                    ],
                    dim=1,
                )
            else:
                image, H, W = None, *args.sampling_image_size

            text_cast = [text]
            mp_size = mpu.get_model_parallel_world_size()
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast_object_list(
                text_cast, src=src, group=mpu.get_model_parallel_group()
            )
            text = text_cast[0]

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }
            # batch, batch_uc = get_batch(
            #     get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            # )
            conditioner_keys = list(
                set([x.input_key for x in model.conditioner.embedders])
            )
            batch, batch_uc = get_batch(conditioner_keys, value_dict, num_samples, T=T)
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )
            for key in c:
                if key != "crossattn":
                    c[key], uc[key] = map(
                        lambda y: y[key][: math.prod(num_samples)].to("cuda"), (c, uc)
                    )
            if args.image2video:
                c["concat"] = uc["concat"] = image

            for index in range(args.batch_size):
                shape = (T, C, H, W) if args.image2video else (T, C, H // 8, W // 8)
                samples_z = (
                    sample_func(c, uc=uc, batch_size=1, shape=shape)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )

                # save_path = os.path.join(
                #     args.output_dir, f"{cnt}_{text.replace(' ', '_').replace('/', '')[:120]}", str(index)
                # )

                save_path = os.path.join(
                    args.output_dir, f"{current_time}-cogvideox1.5"
                )
                os.makedirs(save_path, exist_ok=True)

                if args.only_save_latents:
                    torch.save(
                        samples_z / model.scale_factor,
                        os.path.join(save_path, "latent.pt"),
                    )
                    with open(os.path.join(save_path, "text.txt"), "w") as f:
                        f.write(text)
                else:
                    samples_x = (
                        torch.clamp(
                            (
                                model.decode_first_stage(samples_z).permute(
                                    0, 2, 1, 3, 4
                                )
                                + 1.0
                            )
                            / 2.0,
                            0.0,
                            1.0,
                        )
                        .to(torch.float32)
                        .cpu()
                    )
                    if mpu.get_model_parallel_rank() == 0:
                        save_video_as_grid_and_mp4(
                            samples_x, save_path, fps=args.sampling_fps, key=cnt
                        )


if __name__ == "__main__":
    args = getArgs()
    main(args, model_cls=SATVideoDiffusionEngine)
