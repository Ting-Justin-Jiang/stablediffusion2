"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
"""
DEBUG_MODE: bool = True
import torch

import math
from .merge import *
from typing import Type, Dict, Any, Tuple, Callable
from .utils import isinstance_str, init_generator


class CacheBus:
    """A Bus class for overall control."""
    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.feature_map_broadcast = None
        self.index = index
        self.rand_indices = None
        self.broadcast_range = broadcast_range
        self.step = 0

    # == 1. Cache Merge Operations == #
    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        if self.feature_map is None:
            # x would be the entire feature map during the first cache update
            self.feature_map = x.clone()
            if DEBUG_MODE: print(f"\033[96mCache Push\033[0m: Initial push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        else:
            # x would be the dst (updated) tokens during subsequent cache updates
            self.feature_map.scatter_(dim=-2, index=index, src=x.clone())
            if DEBUG_MODE: print(f"\033[96mCache Push\033[0m: Push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")

    def pop(self, index: torch.Tensor) -> torch.Tensor:
        # Retrieve the src tokens from the cached feature map
        x = torch.gather(self.feature_map, dim=-2, index=index)
        if DEBUG_MODE: print(f"\033[96mCache Pop\033[0m: Pop x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        return x

    # == 2. Broadcast Operations == #
    def save(self, x: torch.Tensor) -> None:
        self.feature_map_broadcast = x.clone()

    def broadcast(self) -> torch.Tensor:
        if self.feature_map_broadcast is None:
            raise RuntimeError
        else:
            return self.feature_map_broadcast

    def should_save(self, broadcast_start: int) -> bool:
        if (self.step - broadcast_start) % self.broadcast_range == 0:
            if DEBUG_MODE: print(f"\033[96mBroadcast\033[0m: Save at step: {self.step} cache index: \033[91m{self.index}\033[0m")
            return True  # Save at this step
        else:
            if DEBUG_MODE: print(f"\033[96mBroadcast\033[0m: Broadcast at step: {self.step} cache index: \033[91m{self.index}\033[0m")
            return False # Broadcast at this step


def compute_merge(x: torch.Tensor, mode:str, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        # retrieve semi-random merging schedule
        if mode == 'cache_merge':
            if cache.rand_indices is None:
                cache.rand_indices = cache.cache_bus.rand_indices[cache.index].copy()
            rand_indices = cache.rand_indices
        else:
            rand_indices = None

        # the function defines the indices to merge and unmerge
        m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, tome_info,
                                              no_rand=not use_rand, unmerge_mode=mode,
                                              cache=cache, rand_indices=rand_indices, generator=args["generator"],)
    else:
        m, u = (do_nothing, do_nothing)

    return m, u


def make_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _mode = mode

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, u_a = compute_merge(x, self._mode, self._tome_info, self._cache)

            x = u_a(self.attn1(m_a(self.norm1(x), prune=True), context=context if self.disable_self_attn else None)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x

            self._cache.step += 1
            return x

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def generate_semi_random_indices(sy: int, sx: int, h: int, w: int, steps: int) -> list:
    """
    Generates a semi-random merging schedule given the grid size.
    """
    hsy, wsx = h // sy, w // sx
    cycle_length = sy * sx
    num_cycles = -(-steps // cycle_length)

    num_positions = hsy * wsx

    # Generate random permutations for all positions
    random_numbers = torch.rand(num_positions, num_cycles * cycle_length)
    indices = random_numbers.argsort(dim=1)
    indices = indices[:, :steps] % cycle_length  # Map indices to [0, cycle_length - 1]

    # Reshape to (hsy, wsx, steps)
    indices = indices.view(hsy, wsx, steps)

    rand_idx = [indices[:, :, step].unsqueeze(-1) for step in range(steps)]
    return rand_idx


def reset_cache(model: torch.nn.Module):
    diffusion_model = model.model.diffusion_model
    bus = diffusion_model._bus
    index = 0
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "ToMeBlock"):
            module._cache = Cache(cache_bus=bus, index=index)
            index += 1
    print(f"Reset cache for {index} BasicTransformerBlock")
    return model


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        mode: str = "token_merge",
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,

        latent_size: Tuple[int, int] = (96, 96),
        merge_step: Tuple[int, int] = (1, 49),
        cache_step: Tuple[int, int] = (1, 49),
        push_unmerged: bool = True,
):
    # == merging preparation ==
    global DEBUG_MODE
    if DEBUG_MODE: print('Start with \033[95mDEBUG\033[0m mode')
    print('\033[94mApplying Token Merging\033[0m')

    cache_start = max(cache_step[0], 1)  # Make sure the first step is token merging to avoid cache access

    # Make sure the module is not currently patched
    remove_patch(model)

    if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
        # Provided model not supported
        raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
    diffusion_model = model.model.diffusion_model

    print(
        "\033[96mArguments:\033[0m\n"
        f"ratio: {ratio}\n"
        f"max_downsample: {max_downsample}\n"
        f"mode: {mode}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"latent_size: {latent_size}\n"
        f"merge_step: {merge_step}\n"
        f"cache_step: {cache_start, cache_step[-1]}\n"
        f"push_unmerged: {push_unmerged}\n"
    )

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "mode": mode,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,

            "latent_size": latent_size,
            "merge_start": merge_step[0],
            "merge_end": merge_step[-1],
            "cache_start": cache_start,
            "cache_end": cache_step[-1],
            "push_unmerged": push_unmerged
        }
    }

    hook_tome_model(diffusion_model)

    diffusion_model._bus = CacheBus()
    index = 0
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_tome_block(module.__class__, mode=mode)
            module._tome_info = diffusion_model._tome_info
            module._cache = Cache(cache_bus=diffusion_model._bus, index=index)
            rand_indices = generate_semi_random_indices(module._tome_info["args"]['sy'],
                                                        module._tome_info["args"]['sx'],
                                                        latent_size[0], latent_size[1], steps=50)
            module._cache.cache_bus.rand_indices[module._cache.index] = rand_indices
            index += 1

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

    print(f"Applied merging patch for BasicTransformerBlock")
    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model