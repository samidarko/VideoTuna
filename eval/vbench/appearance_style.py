import json
import os

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from vbench.utils import (
    clip_transform,
    clip_transform_Image,
    load_dimension_info,
    load_video,
    read_frames_decord_by_fps,
)


def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template = f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features
    return text_features


def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames, test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat


def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels


def appearance_style(clip_model, video_dict, device, sample="rand"):
    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform_Image(224)
    for info in tqdm(video_dict):
        if "auxiliary_info" not in info:
            raise "Auxiliary info is not in json, please check your json."
        query = info["auxiliary_info"]["appearance_style"]
        text = clip.tokenize([query]).to(device)
        video_list = info["video_list"]
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                video_arrays = load_video(video_path, return_tensor=False)
                images = [Image.fromarray(i) for i in video_arrays]
                for image in images:
                    image = image_transform(image)
                    image = image.to(device)
                    logits_per_image, logits_per_text = clip_model(
                        image.unsqueeze(0), text
                    )
                    cur_sim = float(logits_per_text[0][0].cpu())
                    cur_sim = cur_sim / 100
                    cur_video.append(cur_sim)
                    sim += cur_sim
                    cnt += 1
                video_sim = np.mean(cur_video)
                video_results.append(
                    {
                        "video_path": video_path,
                        "video_results": video_sim,
                        "frame_results": cur_video,
                    }
                )
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_appearance_style(json_dir, device, submodules_list, **kwargs):
    clip_model, preprocess = clip.load(device=device, **submodules_list)
    _, video_dict = load_dimension_info(
        json_dir, dimension="appearance_style", lang="en"
    )
    all_results, video_results = appearance_style(clip_model, video_dict, device)
    return all_results, video_results
