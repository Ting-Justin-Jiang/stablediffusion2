"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
"""
DEBUG_MODE: bool = True
import torch
import numpy as np
import math
from .merge import *
from .utils import isinstance_str, init_generator


class CacheBus:
    """A Bus class for overall control."""
    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx
        # self.model_outputs = {}
        # self.model_outputs_change = {}

        self.prev_fm = None
        self.prev_prev_fm = None
        self.temporal_score = None
        self.step = 1  # todo untested, just because dpm++ 2M starts from 2

        # save functions calculated for given step
        self.m_a = None # todo untested
        self.u_a = None

        # indicator for re-calculation
        self.ind_step = None # todo untested
        self.ind_dim = None



class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus

        self.feature_map = None
        self.feature_map_mlp = None

        self.index = index
        self.rand_indices = None
        self.step = 0

    # == 1. Cache Merge Operations == #
    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        if index is None:
            # x would be the entire feature map during the first cache update
            self.feature_map = x
            if DEBUG_MODE: print(f"\033[96mCache Push\033[0m: Initial push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        else:
            # x would be the dst (updated) tokens during subsequent cache updates
            self.feature_map.scatter_(dim=-2, index=index, src=x)
            if DEBUG_MODE: print(f"\033[96mCache Push\033[0m: Push x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")

    def pop(self, index: torch.Tensor) -> torch.Tensor:
        # Retrieve the src tokens from the cached feature map
        x = torch.gather(self.feature_map, dim=-2, index=index)
        if DEBUG_MODE: print(f"\033[96mCache Pop\033[0m: Pop x: \033[95m{x.shape}\033[0m to cache index: \033[91m{self.index}\033[0m")
        return x



def compute_prune(x: torch.Tensor, mode:str, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
    """
    Optimized to avoid re-calculation of pruning and reconstruction function
    """

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if (args['deep_cache'] and (downsample <= args["max_downsample"] and (cache.index in (0, 13, 14)))) or \
            (not args['deep_cache'] and downsample <= args["max_downsample"]):
        if cache.cache_bus.ind_step == cache.step:
            m, u = cache.cache_bus.m_a, cache.cache_bus.u_a

        else:
            # Update indicator
            cache.cache_bus.ind_step = cache.step

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
            if mode == 'cache_merge_deprecated':
                if cache.rand_indices is None:
                    cache.rand_indices = cache.cache_bus.rand_indices[cache.index].copy()
                rand_indices = cache.rand_indices
            else:
                rand_indices = None

            # the function defines the indices to prune
            m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, tome_info,
                                                    no_rand=not use_rand, unmerge_mode=mode,
                                                    cache=cache, rand_indices=rand_indices,
                                                    generator=args["generator"], )

            cache.cache_bus.m_a, cache.cache_bus.u_a= m, u

    else:
        m, u = (do_nothing, do_nothing)

    return m, u



def patch_solver(solver_class):
    class PatchedDPMSolverMultistepScheduler(solver_class):
        def multistep_dpm_solver_second_order_update(
            self,
            model_output_list: List[torch.FloatTensor],
            *args,
            sample: torch.FloatTensor = None,
            noise: Optional[torch.FloatTensor] = None,
            **kwargs,
        ) -> torch.FloatTensor:
            if sample is None:
                if len(args) > 2:
                    sample = args[2]
                else:
                    raise ValueError(" missing `sample` as a required keyward argument")

            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )

            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
            lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

            m0, m1 = model_output_list[-1], model_output_list[-2]

            h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
            r0 = h_0 / h
            D0, D1 = m0, (1.0 / r0) * (m0 - m1)
            if self.config.algorithm_type == "dpmsolver++":
                # See https://arxiv.org/abs/2211.01095 for detailed derivations
                if self.config.solver_type == "midpoint":
                    x_t = (
                        (sigma_t / sigma_s0) * sample
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                    )
                elif self.config.solver_type == "heun":
                    x_t = (
                        (sigma_t / sigma_s0) * sample
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                        + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                    )
                else:
                    raise RuntimeError
            else:
                raise RuntimeError

            # # logging 1
            # feature_map = np.array(m0[0].unsqueeze(dim=0).cpu(), dtype=np.float32).astype(np.float16)
            # self._cache_bus.model_outputs[self._cache_bus.step] = feature_map

            choice = "second_order"

            if choice == "third_order":
                if self._cache_bus.step == 1: # the first time this code is reached
                    self._cache_bus.prev_fm = m1.clone()
                    score = None

                elif self._cache_bus.step == 2: # the second time this code is reached
                    self._cache_bus.prev_prev_fm = self._cache_bus.prev_fm.clone()
                    self._cache_bus.prev_fm = m1.clone()
                    score = None

                else:
                    cur_diff = (m0 - m1).abs().mean(dim=1)
                    prev_diff = (m1 - self._cache_bus.prev_fm).abs().mean(dim=1)
                    prev_prev_diff = (self._cache_bus.prev_fm - self._cache_bus.prev_prev_fm).abs().mean(dim=1)

                    diff = abs((cur_diff + prev_prev_diff) / 2 - prev_diff)
                    score = (diff > 0.75 * prev_diff).float()

                    print(int(score.sum().item()))

                    self._cache_bus.temporal_score = score

                    self._cache_bus.prev_prev_fm = self._cache_bus.prev_fm.clone()
                    self._cache_bus.prev_fm = m1.clone()

            elif choice == "second_order":
                if self._cache_bus.step == 1:  # the first time this code is reached
                    self._cache_bus.prev_fm = m1.clone()
                    score = None

                else:
                    cur_diff = (m0 - m1).abs().mean(dim=1)
                    prev_diff = (m1 - self._cache_bus.prev_fm).abs().mean(dim=1)

                    diff = abs(cur_diff - prev_diff)
                    # score = (diff >= 0.15 * m1.abs().mean()).float() # here the metric is a single real number

                    # Calculate the dynamic threshold value
                    threshold_value = 0.15 * m1.abs().mean()
                    threshold_num = 1024*4

                    diff_flat = diff.view(-1)
                    mask = diff_flat >= threshold_value
                    num_exceeds = mask.sum().item()
                    score_flat = torch.zeros_like(diff_flat)

                    if num_exceeds > threshold_num:
                        # If more elements exceed the threshold than allowed, select the top 'threshold' elements
                        topk_values, topk_indices = torch.topk(diff_flat, threshold_num, largest=True, sorted=False)
                        score_flat[topk_indices] = 1.0
                    else:
                        # If fewer elements exceed the threshold, set all of them to 1
                        score_flat[mask] = 1.0

                    score = score_flat.view_as(diff)

                    self._cache_bus.temporal_score = score
                    self._cache_bus.prev_fm = m1.clone()
            else:
                raise NotImplementedError

            # # logging 2
            # if score is not None:
            #     score = score.expand([1, 4, 96, 96])
            #     m0 = m0 + score * 2.5
            # feature_map = np.array(m0[0].unsqueeze(dim=0).cpu(), dtype=np.float32).astype(np.float16)
            # self._cache_bus.model_outputs_change[self._cache_bus.step] = feature_map

            self._cache_bus.step += 1
            return x_t

    return PatchedDPMSolverMultistepScheduler



def make_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _mode = mode

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, u_a = compute_prune(x, self._mode, self._tome_info, self._cache)

            x = u_a(self.attn1(m_a(self.norm1(x), prune=self._tome_info['args']['prune']), context=context if self.disable_self_attn else None)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x

            self._cache.step += 1
            return x

    return ToMeBlock



def make_diffusers_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _mode = mode

        def forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=None,
                cross_attention_kwargs=None,
                class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, u_a = compute_prune(hidden_states, self._mode, self._tome_info, self._cache)
            hidden_states = m_a(hidden_states, prune=self._tome_info['args']['prune'], cache=self._cache)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states


            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states

            hidden_states = u_a(hidden_states, cache=self._cache)

            self._cache.step += 1

            return hidden_states

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
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    if not is_diffusers:
        diffusion_model = model.model.diffusion_model
    else:
        diffusion_model = model.unet if hasattr(model, "unet") else model

    # reset bus
    bus = diffusion_model._bus
    bus.prev_prev_fm = None
    bus.temporal_score = None
    bus.step = 1

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
        prune: bool = "true",
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,

        latent_size: Tuple[int, int] = (96, 96),
        merge_step: Tuple[int, int] = (1, 49),
        cache_step: Tuple[int, int] = (1, 49),
        push_unmerged: bool = True,

        deep_cache: bool = True
):
    # == merging preparation ==
    global DEBUG_MODE
    if DEBUG_MODE: print('Start with \033[95mDEBUG\033[0m mode')
    print('\033[94mApplying Token Merging\033[0m')

    cache_start = max(cache_step[0], 1)  # Make sure to avoid cache access in the first step

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model
    solver = model.scheduler

    print(
        "\033[96mArguments:\033[0m\n"
        f"ratio: {ratio}\n"
        f"prune: {prune}\n"
        f"max_downsample: {max_downsample}\n"
        f"mode: {mode}\n"
        f"sx: {sx}, sy: {sy}\n"
        f"use_rand: {use_rand}\n"
        f"latent_size: {latent_size}\n"
        f"merge_step: {merge_step}\n"
        f"cache_step: {cache_start, cache_step[-1]}\n"
        f"push_unmerged: {push_unmerged}\n"
        f"deep_cahce: {deep_cache}"
    )

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "mode": mode,
            "prune": prune,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,

            "latent_size": latent_size,
            "merge_start": merge_step[0],
            "merge_end": merge_step[-1],
            "cache_start": cache_start,
            "cache_end": cache_step[-1],
            "push_unmerged": push_unmerged,

            "deep_cache": deep_cache
        }
    }

    hook_tome_model(diffusion_model)

    diffusion_model._bus = CacheBus()

    solver.__class__ = patch_solver(solver.__class__)
    solver._cache_bus = diffusion_model._bus

    index = 0
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__, mode=mode)
            module._tome_info = diffusion_model._tome_info
            module._cache = Cache(cache_bus=diffusion_model._bus, index=index)
            rand_indices = generate_semi_random_indices(module._tome_info["args"]['sy'],
                                                        module._tome_info["args"]['sx'],
                                                        latent_size[0], latent_size[1], steps=60)
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



def get_logged_feature_maps(model: torch.nn.Module, file_name: str = "outputs/model_outputs.npz"):
    logging.debug(f"\033[96mLogging Feature Map\033[0m")
    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in model._bus.model_outputs.items()}
    np.savez(file_name, **numpy_feature_maps)

    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in model._bus.model_outputs_change.items()}
    np.savez("outputs/model_outputs_change.npz", **numpy_feature_maps)



# class Cache:
#     def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
#         self.cache_bus = cache_bus
#
#         self.feature_map = None
#         self.hsy = None
#         self.wsx = None
#         self.sy_sx = None
#
#         self.index = index
#         self.rand_indices = None
#         self.step = 0
#
#     # == 1. Cache Merge Operations == #
#     def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
#         """
#         Pushes preserved patches to the cache.
#
#         Args:
#             x (torch.Tensor): Tensor of preserved patches with shape [B, N_p, C],
#                               where N_p = number of preserved patches * sy * sx.
#             index (torch.Tensor): Tensor of patch indices with shape [num_preserved_patches, 3],
#                                    where each row is [Batch_idx, hsy_idx, wsx_idx].
#         """
#         if self.feature_map is None:
#             # Initialize the feature_map tensor to store preserved patches
#             self.feature_map = x
#
#             if DEBUG_MODE:
#                 print(f"\033[96mCache Push\033[0m: Initialized feature_map with shape {self.feature_map.shape}")
#
#             return
#
#         if self.hsy is None and index is not None:
#             # Extract dimensions from index
#             B = self.feature_map.shape[0]
#             hsy = index[:, 1].max().item() + 1  # Number of patches along height
#             wsx = index[:, 2].max().item() + 1  # Number of patches along width
#             sy_sx = self.feature_map.shape[1] // (hsy * wsx)  # Tokens per patch (sy * sx)
#
#             self.hsy = hsy
#             self.wsx = wsx
#             self.sy_sx = sy_sx
#
#             self.feature_map = self.feature_map.reshape([B, hsy, wsx, sy_sx, -1])
#
#         # Extract batch and patch indices
#         hsy_idx = index[:, 1]    # Shape: [num_preserved_patches]
#         wsx_idx = index[:, 2]    # Shape: [num_preserved_patches]
#         num_preserved_patches = hsy_idx.shape[0]
#
#         # Validate indices
#         if (hsy_idx.max() >= self.hsy) or (wsx_idx.max() >= self.wsx):
#             raise IndexError("Patch indices exceed initialized feature_map dimensions.")
#
#         # Assign preserved patches to the feature_map
#         self.feature_map[:, hsy_idx, wsx_idx, :, :] = x.view(2, num_preserved_patches, self.sy_sx, -1)
#
#         if DEBUG_MODE:
#             print(f"\033[96mCache Push\033[0m: Pushed patches to cache index: {self.index}")
#
#         return
#
#     def pop(self, index: torch.Tensor) -> torch.Tensor:
#         """
#         Retrieves pruned patches from the cache.
#
#         Args:
#             index (torch.Tensor): Tensor of patch indices with shape [num_pruned_patches, 3],
#                                    where each row is [Batch_idx, hsy_idx, wsx_idx].
#
#         Returns:
#             torch.Tensor: Tensor of retrieved patches with shape [B, num_pruned_patches * sy * sx, C].
#         """
#         if self.feature_map is None:
#             raise RuntimeError("Feature map is empty. Cannot pop from cache.")
#
#         # Extract batch and patch indices
#         hsy_idx = index[:, 1]    # Shape: [num_pruned_patches]
#         wsx_idx = index[:, 2]    # Shape: [num_pruned_patches]
#
#         # Validate indices
#         if (hsy_idx.max() >= self.hsy) or (wsx_idx.max() >= self.wsx):
#             raise IndexError("Patch indices exceed initialized feature_map dimensions.")
#
#         # Retrieve pruned patches from the feature_map
#         pruned_patches = self.feature_map[:, hsy_idx, wsx_idx, :, :]  # Shape: [B, sy*sx, C]
#
#         if DEBUG_MODE:
#             print(f"\033[96mCache Pop\033[0m: Popped patches from cache index: {self.index}")
#
#         return pruned_patches
