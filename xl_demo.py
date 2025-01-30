import argparse
import logging
import math
import time
import torch
import lpips

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from torchvision.utils import save_image

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torchvision.transforms as T

from ldm.tome import patch


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default="A photograph of a cute racoon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt

    baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    baseline_pipe.scheduler = DPMSolverMultistepScheduler.from_config(baseline_pipe.scheduler.config)

    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = baseline_pipe(prompt, num_inference_steps=50, output_type='pt').images

    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)

    ori_output = baseline_pipe(prompt, num_inference_steps=50, output_type='pt').images
    baseline_use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(baseline_use_time))

    del baseline_pipe
    torch.cuda.empty_cache()

    # CAP
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    patch.apply_patch(pipe,
                      ratio=0.99,
                      mode="cache_merge",
                      sx=3, sy=3,
                      max_downsample=1,
                      acc_range=(9, 45),
                      push_unmerged=True,
                      prune=True,

                      max_fix=1024 * 8,
                      max_interval=3)


    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = pipe(prompt, num_inference_steps=50, output_type='pt').images
        patch.reset_cache(pipe)

    logging.info("Running CAP...")
    set_random_seed(seed)
    start_time = time.time()

    cap_output = pipe(prompt, num_inference_step=50, output_type='pt').images
    use_time = time.time() - start_time
    logging.info("CAP: {:.2f} seconds".format(use_time))

    logging.info("Baseline: {:.2f} seconds. CAP: {:.2f} seconds".format(baseline_use_time, use_time))
    save_image([ori_output[0], cap_output[0]], "output.png")
    logging.info("Saved to output.png. Done!")

    print(pipe.unet._cache_bus.rel_momentum_list)
    print(pipe.unet._cache_bus.skipping_path)

    print("Evaluating LPIPS")
    p_r = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in ori_output]).to('cuda')

    p_o = torch.stack([T.Compose([
        T.Normalize((0.5,), (0.5,))
    ])(img) for img in cap_output]).to('cuda')

    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
    d = loss_fn_alex(p_r, p_o)
    print(f"LPIPS: {d.item()}")