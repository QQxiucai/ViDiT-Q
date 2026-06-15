import argparse
import os

import torch
from mmengine.config import Config
from safetensors import safe_open

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, build_module
from opensora.utils.inference_utils import (
    append_score_to_prompts,
    load_prompts,
    prepare_multi_resolution_info,
)
from qdiff.utils import seed_everything


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", default="./precomputed_text_embeds.pth")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    seed_everything(cfg.get("seed", 1024))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if cfg.get("dtype", "fp16") == "fp16" else torch.bfloat16

    prompts = load_prompts(cfg.prompt_path, cfg.get("start_index", 0), cfg.get("end_index", None))
    prompts = [append_score_to_prompts([prompt], aes=cfg.get("aes", None), flow=cfg.get("flow", None))[0] for prompt in prompts]
    prompts = [text_preprocessing(prompt) for prompt in prompts]

    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    model_args = text_encoder.encode(prompts)

    model_file = os.path.join(cfg.model.from_pretrained, "model.safetensors")
    with safe_open(model_file, framework="pt", device=device) as f:
        y_embedding = f.get_tensor("y_embedder.y_embedding").to(dtype=model_args["y"].dtype)
    y_null = y_embedding[None].repeat(len(prompts), 1, 1)[:, None]
    model_args["y"] = torch.cat([model_args["y"], y_null], dim=0)

    image_size = cfg.get("image_size", None)
    if image_size is None:
        image_size = get_image_size(cfg.resolution, cfg.aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)
    model_args.update(
        prepare_multi_resolution_info(
            cfg.get("multi_resolution", None),
            len(prompts),
            image_size,
            num_frames,
            cfg.fps,
            device,
            dtype,
        )
    )

    torch.save({"model_args": to_cpu(model_args), "prompts": prompts}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

