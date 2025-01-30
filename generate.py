import time
import argparse
import numpy as np
import random
import math
import lpips

import os
from tqdm import tqdm

import torch
from datasets import load_dataset

from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.transforms.functional import to_pil_image

from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

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

    if args.model == "stabilityai/stable-diffusion-2-1":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        if args.solver == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.solver == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        max_downsample = 1
        image_size = 768

    elif args.model == "stabilityai/stable-diffusion-xl-base-1.0":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        if args.solver == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.solver == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        max_downsample = 2
        image_size = 1024

    else: raise NotImplementedError

    if args.method == 'cap':
        patch.apply_patch(pipe,
                          ratio=0.99,
                          mode="cache_merge",
                          sx=3, sy=3,
                          max_downsample=max_downsample,
                          acc_range=(9, 45),
                          push_unmerged=True,
                          prune=True,

                          max_fix=args.max_fix,
                          max_interval=args.max_interval)


    # Initialize metric
    inception = InceptionScore().to("cuda:0")
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda:0")

    output_dir = args.experiment_folder
    os.makedirs(output_dir, exist_ok=True)

    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    global_image_index = 0  # Tracks unique image indices across batches
    use_time = 0

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)]

        set_random_seed(args.seed)
        start_time = time.time()
        if args.method != "deep_cache":
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
        use_time += round(time.time() - start_time, 2)
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

        if args.method == 'cap':
            patch.reset_cache(pipe)

    print(f"Done: use_time = {use_time}")
    IS = inception.compute()
    CLIP = clip.compute()
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")

    fid_value = calculate_fid_given_paths(
        [args.target_folder, args.experiment_folder],
        1,
        "cuda:0",
        dims=image_size,
        num_workers=8,
    )
    print(f"FID: {fid_value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # == Sampling setup ==
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-xl-base-1.0') #stabilityai/stable-diffusion-xl-base-1.0  #stabilityai/stable-diffusion-2-1
    parser.add_argument("--dataset", type=str, default="coco2017")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-fid-samples", type=int, default=30)
    parser.add_argument('--experiment-folder', type=str, default='samples/inference/cap')
    parser.add_argument('--target-folder', type=str, default='samples/data/val2017')
    parser.add_argument("--solver", type=str, choices=["euler", "dpm"], default="euler")

    # == Acceleration Setup ==
    parser.add_argument("--method", type=str, choices=["original", "deep_cache", "cap"], default="cap")

    parser.add_argument("--max-fix", type=int, default=1024 * 10)
    parser.add_argument("--max-interval", type=int, default=3)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)