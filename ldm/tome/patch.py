import torch
import math
from .merge import *
from typing import Type, Dict, Any, Tuple, Callable
from tomesd.utils import isinstance_str, init_generator


class CacheBus:
    """A Bus class for efficient communication between blocks at a given timestep."""
    def __init__(self):
        self.feature_maps = {} # key: index, value: feature_maps
        self.rand_indices = {} # key: downsampling (for partial attention) / block_index, value: rand_idx
        self.similarity_scores = {} # key:  downsampling, value: similarity_scores


class Cache:
    def __init__(self, cache_bus: CacheBus, index: int):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.index = index
        self.step = 0

    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        if self.feature_map is None:
            # x would be the entire feature map during the first cache update
            self.feature_map = x.clone()
            self.cache_bus.feature_maps[self.index] = self.feature_map # this will add feature maps to corresponding indices
        else:
            # x would be the dst (updated) tokens during subsequent cache updates
            self.feature_map.scatter_(dim=-2, index=index, src=x.clone())
            self.cache_bus.feature_maps[self.index].scatter_(dim=-2, index=index, src=x.clone())

    def pop(self, index: torch.Tensor) -> torch.Tensor:
        # Retrieve the src tokens from the cached feature map
        return torch.gather(self.feature_map, dim=-2, index=index)


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
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
        if tome_info['args']['semi_rand_schedule']:
            if not tome_info['args']['partial_attention']:
                if cache.index not in cache.cache_bus.rand_indices:
                    rand_indices = generate_semi_random_indices(tome_info["args"]['sx'], tome_info["args"]['sy'], h, w, steps=128)
                    cache.cache_bus.rand_indices[cache.index] = rand_indices
                else:
                    rand_indices = cache.cache_bus.rand_indices[cache.index]
            else:
                if downsample not in cache.cache_bus.rand_indices:
                    rand_indices = generate_semi_random_indices(tome_info["args"]['sx'], tome_info["args"]['sy'], h, w, steps=128)
                    cache.cache_bus.rand_indices[downsample] = rand_indices
                else:
                    rand_indices = cache.cache_bus.rand_indices[downsample]
        else:
            rand_indices = None

        # the function defines the indices to merge and unmerge
        m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, tome_info,
                                                      no_rand=not use_rand, generator=args["generator"],
                                                      cache=cache, rand_indices=rand_indices)
        message = f"token merging operates at downsample: \033[91m{downsample}\033[0m, original x shape: \033[91m{x.shape}\033[0m, merged: \033[91m{r}\033[0m"
    else:
        m, u = (do_nothing, do_nothing)
        message = "token merging does nothing"

    m_a, u_a = (m, u) if args["merge_attn"] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info, self._cache)

            # This is where the meat of the computation happens
            # self attention
            x_a = m_a(self.norm1(x))
            x_a = self.attn1(x_a, context=context if self.disable_self_attn else None)
            x = u_a(x_a) + x

            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            self._cache.step += 1

            return x

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def generate_semi_random_indices(sy, sx, h, w, steps):
    """
    generates a semi-random merging schedule given the grid size
    """
    hsy, wsx = h // sy, w // sx
    cycle_length = sy * sx
    num_cycles = (steps + cycle_length - 1) // cycle_length

    full_sequence = []

    for _ in range(hsy * wsx):
        sequence = torch.cat([
            torch.randperm(cycle_length)
            for _ in range(num_cycles)
        ])
        full_sequence.append(sequence[:steps])

    full_sequence = torch.stack(full_sequence).to(torch.int64)
    rand_idx = full_sequence.reshape(hsy, wsx, steps).permute(2, 0, 1).unsqueeze(-1)
    return rand_idx


def reset_cache(model: torch.nn.Module):
    diffusion_model = model.model.diffusion_model
    bus = CacheBus()
    index = 0
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "ToMeBlock"):
            module._cache = Cache(cache_bus=bus, index=index)
            index += 1
    print(f"Reset cache for {index} BasicTransformerBlock")
    return model


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,

        # == Cache Merge ablation arguments == #
        semi_rand_schedule: bool = False,
        unmerge_residual: bool = False,
        cache_similarity: bool = False,
        partial_attention: bool = False,
):

    if not semi_rand_schedule:
        assert partial_attention is False, "Cannot apply partial attention without merging scheduler"
    if cache_similarity:
        assert partial_attention is True, "Cannot cache similarity without partial attention"

    # Make sure the module is not currently patched
    remove_patch(model)

    if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
        # Provided model not supported
        raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
    diffusion_model = model.model.diffusion_model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,

            # == Cache Merge ablation arguments == #
            "semi_rand_schedule": semi_rand_schedule,
            "unmerge_residual": unmerge_residual,
            "cache_similarity": cache_similarity,
            "partial_attention": partial_attention
        }
    }

    hook_tome_model(diffusion_model)

    bus = CacheBus()
    index = 0
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_tome_block(module.__class__)
            module._tome_info = diffusion_model._tome_info
            module._cache = Cache(cache_bus=bus, index=index)

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