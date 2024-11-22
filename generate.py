import time
import argparse
import numpy as np
import random
import math

import os
from tqdm import tqdm

import torch
from datasets import load_dataset

from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.transforms.functional import to_pil_image

from ldm.tome import patch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    if args.dataset == 'parti':
        prompts = load_dataset("nateraw/parti-prompts", split="train")
    elif args.dataset == 'coco2017':
        dataset = load_dataset("phiyodr/coco2017")
        prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
    else:
        raise NotImplementedError

    prompts = prompts[:args.num_fid_samples]

    if args.original:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        if args.merge_ratio > 0.0:
            patch.apply_patch(pipe,
                              ratio=args.merge_ratio,
                              mode=args.merge_mode,
                              prune=args.prune,
                              sx=2, sy=2,
                              max_downsample=1,
                              latent_size=(2 * math.ceil(768 / 16),
                                           2 * math.ceil(768 / 16)),
                              merge_step=args.merge_step,
                              cache_step=args.cache_step,
                              push_unmerged=args.push_unmerged,
                              deep_cache=False)

    else:
        from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda:0")
        if args.merge_ratio > 0.0:
            patch.apply_patch(pipe,
                              ratio=args.merge_ratio,
                              mode=args.merge_mode,
                              prune=args.prune,
                              sx=2, sy=2,
                              max_downsample=1,
                              latent_size=(2 * math.ceil(768 / 16),
                                           2 * math.ceil(768 / 16)),
                              merge_step=args.merge_step,
                              cache_step=args.cache_step,
                              push_unmerged=args.push_unmerged,
                              deep_cache=True)


    # Initialize metric
    inception = InceptionScore().to("cuda:0")
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda:0")

    output_dir = args.experiment_folder
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    global_image_index = 0  # Tracks unique image indices across batches

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)]

        set_random_seed(args.seed)
        if args.original or args.bk:
            pipe_output = pipe(
                sample_prompts, output_type='np', return_dict=True,
                num_inference_steps=args.steps
            )
        else:
            pipe_output = pipe(
                sample_prompts, num_inference_steps=args.steps,
                cache_interval=args.update_interval,
                cache_layer_id=args.layer, cache_block_id=args.block,
                uniform=args.uniform, pow=args.pow, center=args.center,
                output_type='np', return_dict=True
            )
        images = pipe_output.images

        # Inception Score & CLIP
        torch_images = torch.Tensor(images * 255).byte().permute(0, 3, 1, 2).contiguous()
        torch_images = torch.nn.functional.interpolate(
            torch_images, size=(299, 299), mode="bilinear", align_corners=False
        ).to("cuda:0")
        inception.update(torch_images)
        clip.update(torch_images, sample_prompts)

        for image in images:
            image = to_pil_image((image * 255).astype(np.uint8))  # Convert to PIL image
            image.save(f"{output_dir}/{global_image_index}.jpg")  # Use global index
            global_image_index += 1

        if args.merge_ratio > 0.0:
            patch.reset_cache(pipe)

    use_time = round(time.time() - start_time, 2)
    print(f"Done: use_time = {use_time}")
    IS = inception.compute()
    CLIP = clip.compute()
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")

    fid_value = calculate_fid_given_paths(
        [args.target_folder, args.experiment_folder],
        1,
        "cuda:0",
        dims=2048,
        num_workers=8,
    )
    print(f"FID: {fid_value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco2017")

    # For choosing baselines. If these two are not set, then it will use DeepCache.
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--bk", type=str, default=None)

    # Hyperparameters for DeepCache
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--update_interval", type=int, default=5)
    parser.add_argument("--uniform", action="store_true", default=True)
    parser.add_argument("--pow", type=float, default=1.4)
    parser.add_argument("--center", type=int, default=15)

    # Hyperparameters for CAP
    parser.add_argument("--merge-ratio", type=float, default=0.7)
    parser.add_argument("--merge-metric", type=str, choices=["k", "x"], default="k")
    parser.add_argument("--merge-mode", type=str, choices=["token_merge", "cache_merge"], default="token_merge")
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, type=bool, default=False)
    parser.add_argument("--merge-cond", action=argparse.BooleanOptionalAction, type=bool, default=False)

    parser.add_argument("--merge-step", type=lambda s: (int(item) for item in s.split(',')), default=(0, 48))
    parser.add_argument("--cache-step", type=lambda s: (int(item) for item in s.split(',')), default=(9, 48))
    parser.add_argument("--push-unmerged", action=argparse.BooleanOptionalAction, type=bool, default=True)


    # Sampling setup
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-fid-samples", type=int, default=20)
    parser.add_argument('--experiment-folder', type=str, default='samples/experiment/token_prune')
    parser.add_argument('--target-folder', type=str, default='samples/data/val2017')

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)