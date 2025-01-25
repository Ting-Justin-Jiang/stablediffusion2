"""
Code adapted from original tomesd: https://github.com/dbolya/tomesd
"""
DEBUG_MODE: bool = True
import torch
import numpy as np
import math
from .merge import *
from .utils import isinstance_str, init_generator

from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, scale_lora_layers, unscale_lora_layers
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class CacheBus:
    """A Bus class for overall control."""

    def __init__(self):
        self.rand_indices = {}  # key: index, value: rand_idx

        # momentum awareness
        self.prev_x0 = None
        self.prev_prev_x0 = None
        self.prev_sample_list = [None, None]
        self.temporal_score = None
        self.step = 0

        # partial calculation of momentum
        self.downsample = 1
        self.b_idx = None

        # save functions calculated for given step
        self.m_a = None  # todo untested
        self.u_a = None

        # indicator for re-calculation
        self.c_step = 1  # todo untested, just because dpm++ 2M starts from 2
        self.ind_step = None  # todo untested

        # model-level accel
        self.prev_momentum = None
        self.last_skip_step = 0  # align with step in cache bus
        self.skip_this_step = False
        self.prev_noise = [None, None]
        self.prev_f = [None, None, None]

        # remove comment to log
        # self.model_outputs = {}
        # self.model_outputs_change = {}
        self.pred_m_m_1 = None
        self.taylor_m_m_1 = None
        self.pred_error_list = []
        self.taylor_error_list = []

        self.abs_momentum_list = []
        self.rel_momentum_list = []
        self.skipping_path = []


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus

        self.feature_map = None
        self.feature_map_mlp = None

        self.index = index
        self.rand_indices = None

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


def compute_prune(x: torch.Tensor, mode: str, tome_info: Dict[str, Any], cache: Cache) -> Tuple[Callable, ...]:
    """
    Optimized to avoid re-calculation of pruning and reconstruction function
    """

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        if cache.cache_bus.ind_step == cache.cache_bus.step:
            m, u = cache.cache_bus.m_a, cache.cache_bus.u_a

        else:  # when reaching a new step
            # Update indicator
            cache.cache_bus.ind_step = cache.cache_bus.step
            cache.cache_bus.downsample = downsample

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

            cache.cache_bus.m_a, cache.cache_bus.u_a = m, u

    else:
        m, u = (do_nothing, do_nothing)

    return m, u


def patch_solver(solver_class):
    class PatchedDPMSolverMultistepScheduler(solver_class):
        def step(
                self,
                model_output: torch.FloatTensor,
                timestep: int,
                sample: torch.FloatTensor,
                generator=None,
                return_dict: bool = True,
        ) -> Union[SchedulerOutput, Tuple]:
            if self.num_inference_steps is None:
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )

            if self.step_index is None:
                self._init_step_index(timestep)

            # Improve numerical stability for small number of steps
            lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
                    self.config.euler_at_final or (self.config.lower_order_final and len(self.timesteps) < 15)
            )
            lower_order_second = (
                    (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(
                self.timesteps) < 15
            )




            # # === TODO: Experiment area, Do NOT write anything new pass this ===
            # # == before we convert everything to data prediction: ==
            # N = self.timesteps.shape[0]
            # beta_n = self.betas[self.timesteps[self.step_index - 1]]
            # sigma_t, sigma_s0, sigma_s1 = (
            #     self.sigmas[self.step_index + 1],
            #     self.sigmas[self.step_index],
            #     self.sigmas[self.step_index - 1],
            # )
            # # CFG to the noise
            # epsilon_0 = self._cache_bus.prev_noise[-1][0] + 5.0 * (
            #         self._cache_bus.prev_noise[-1][1] - self._cache_bus.prev_noise[-1][0])
            #
            # f = (- 0.5 * beta_n * N * sample) + (0.5 * beta_n * N / sigma_s0) * epsilon_0
            #
            # if self._cache_bus.prev_f[0] is not None:
            #     momentum = ((self._cache_bus.prev_f[-1] - self._cache_bus.prev_f[-2]) - (
            #                 self._cache_bus.prev_f[-2] - self._cache_bus.prev_f[-3])).abs() / (f.abs() + 1e-5)
            #     momentum = momentum.mean()
            #
            # for i in range(2):
            #     self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
            # self._cache_bus.prev_f[-1] = f
            #
            # # == now we have a numerical 3-order difference ==
            # # === TODO: Experiment area, Do NOT write anything new pass this ===



            model_output = self.convert_model_output(model_output, sample=sample)

            for i in range(self.config.solver_order - 1):
                self._cache_bus.prev_sample_list[i] = self._cache_bus.prev_sample_list[i + 1]
                self.model_outputs[i] = self.model_outputs[i + 1]

            self.model_outputs[-1] = model_output
            self._cache_bus.prev_sample_list[-1] = sample

            noise = None

            if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
            elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample,
                                                                            noise=noise)
            else:
                prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)

            if self.lower_order_nums < self.config.solver_order:
                self.lower_order_nums += 1

            self._step_index += 1

            if not return_dict:
                return (prev_sample,)

            return SchedulerOutput(prev_sample=prev_sample)

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
            # self._cache_bus.model_outputs[self._cache_bus.c_step] = feature_map

            choice = "second_order"

            if choice == "second_order":
                if self._cache_bus.c_step == 1:  # the first time this code is reached
                    self._cache_bus.prev_x0 = m1.clone()
                    score = None

                else:
                    # get hyperparameters
                    threshold_token = self._cache_bus._tome_info['args']['threshold_token']
                    threshold_map = self._cache_bus._tome_info['args']['threshold_map']
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    # # todo: only for testing, delete later
                    # if self._cache_bus._tome_info['args']['test_skip_path'] and self._cache_bus.step in self._cache_bus._tome_info['args']['test_skip_path']:
                    #     self._cache_bus.skip_this_step = True

                    if self._cache_bus.prev_prev_x0 is not None:

                        if self.step_index == 0:
                            raise RuntimeError("Should start calculating momentum with at lest two latents")

                        N = self.timesteps.shape[0]

                        beta_n = self.betas[self.timesteps[self.step_index - 1]]
                        beta_n_1 = self.betas[self.timesteps[min(N - 1, self.step_index)]]
                        beta_n_m1 = self.betas[self.timesteps[self.step_index - 2]]

                        s_alpha_cumprod_n = self.alpha_t[self.timesteps[self.step_index - 1]]
                        s_alpha_cumprod_n_1 = self.alpha_t[self.timesteps[min(N - 1, self.step_index)]]
                        s_alpha_cumprod_n_m1 = self.alpha_t[self.timesteps[self.step_index - 2]]

                        delta = 1 / N  # step size correlates with number of inference step

                        # CFG to the noise
                        epsilon_0 = self._cache_bus.prev_noise[-1][0] + 5.0 * (
                                    self._cache_bus.prev_noise[-1][1] - self._cache_bus.prev_noise[-1][0])
                        epsilon_1 = self._cache_bus.prev_noise[-2][0] + 5.0 * (
                                    self._cache_bus.prev_noise[-2][1] - self._cache_bus.prev_noise[-2][0])

                        # AM method
                        term_1 = (1 + 0.25 * delta * N * beta_n) * s_alpha_cumprod_n * m0
                        term_2 = 0.25 * delta * N * beta_n_1 * s_alpha_cumprod_n_1 * m1
                        term_3 = ((1 + 0.25 * delta * N * beta_n) * sigma_s0 - sigma_s1 - (delta * N * beta_n) / (4 * sigma_s0)) * epsilon_0
                        term_4 = (0.25 * delta * N * beta_n_1 * sigma_t - (delta * N * beta_n_1) / (4 * sigma_t)) * epsilon_1

                        m_m_1 = (term_1 + term_2 + term_3 + term_4) / s_alpha_cumprod_n_m1

                        # calculate MSE for our prediction
                        if self._cache_bus.pred_m_m_1 is not None:
                            error = ((self._cache_bus.pred_m_m_1 - m0) ** 2).mean()
                            print(f"\n Our Approximation error - : {error}")
                            self._cache_bus.pred_error_list.append(float(error.item()))
                            print(self._cache_bus.pred_error_list)
                        self._cache_bus.pred_m_m_1 = m_m_1.clone()

                        # calculate MSE for taylor prediction
                        taylor_m_m_1 = m0 + (m0 - m1) + 0.5 * (m0 - 2 * m1 + self._cache_bus.prev_x0)
                        if self._cache_bus.taylor_m_m_1 is not None:
                            taylor_error = ((self._cache_bus.taylor_m_m_1 - m0) ** 2).mean()
                            print(f"\n Taylor Approximation error - : {taylor_error}")
                            self._cache_bus.taylor_error_list.append(float(taylor_error.item()))
                            print(self._cache_bus.taylor_error_list)
                        self._cache_bus.taylor_m_m_1 = taylor_m_m_1.clone()



                        # === TODO: Experiment area, Do NOT write anything new pass this ===

                        # # Token-wise Calculation
                        # momentum = (m_m_1 - 3 * m0 + 3 * m1 - self._cache_bus.prev_x0).view(-1)
                        # threshold_value = threshold_token * (m_m_1 - m0).view(-1)
                        #
                        # mask = momentum >= threshold_value
                        # num_exceeds = mask.sum().item()
                        #
                        # self._cache_bus.rel_momentum_list.append(num_exceeds)
                        #
                        # if num_exceeds <= 30000:
                        #     self._cache_bus.skip_this_step = True


                        # Feature-map-wise Calculation
                        momentum = (m_m_1 - 3 * m0 + 3 * m1 - self._cache_bus.prev_x0).mean()
                        abs_momentum = float(momentum.item())
                        rel_momentum = float((momentum.item() / sigma_s0).item())
                        self._cache_bus.abs_momentum_list.append(abs_momentum)
                        self._cache_bus.rel_momentum_list.append(rel_momentum)

                        # if momentum <= - 0.0002 *  sigma_s0:
                        #     self._cache_bus.skip_this_step = True

                        # === TODO: Experiment area, Do NOT write anything new pass this ===



                    if self._cache_bus.skip_this_step:
                        # == In this branch, we proceed feature map-level pruning ==
                        score = None
                        self._cache_bus.b_idx = None

                    else:
                        # == In this branch, we proceed feature-level pruning ==
                        # calculate the momentum based on x0 prediction
                        cur_diff = (m0 - m1).abs().mean(dim=1)
                        prev_diff = (m1 - self._cache_bus.prev_x0).abs().mean(dim=1)
                        momentum = (cur_diff - prev_diff).abs()

                        # Calculate the dynamic threshold value
                        threshold_value = threshold_token * m1.abs().mean()

                        momentum_flat = momentum.view(-1)
                        mask = momentum_flat >= threshold_value
                        num_exceeds = mask.sum().item()
                        score_flat = torch.zeros_like(momentum_flat)

                        print(f"num_exceed {num_exceeds}")

                        if num_exceeds > max_fix:
                            # If more elements exceed the threshold than allowed, select the top 'threshold' elements
                            topk_values, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                            score_flat[topk_indices] = 1.0
                        else:
                            # If fewer elements exceed the threshold, set all of them to 1
                            score_flat[mask] = 1.0

                        score = score_flat.view_as(momentum)

                    self._cache_bus.prev_prev_x0 = self._cache_bus.prev_x0
                    self._cache_bus.prev_x0 = m1.clone()
                    self._cache_bus.temporal_score = score

            else:
                raise NotImplementedError

            # # logging 2
            # if score is not None:
            #     score = score.expand([1, 4, 96, 96])
            #     m0 = m0 + score * 2.5
            # feature_map = np.array(m0[0].unsqueeze(dim=0).cpu(), dtype=np.float32).astype(np.float16)
            # self._cache_bus.model_outputs_change[self._cache_bus.c_step] = feature_map

            self._cache_bus.c_step += 1
            return x_t

    return PatchedDPMSolverMultistepScheduler


def make_tome_block(block_class: Type[torch.nn.Module], mode: str = 'token_merge') -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _mode = mode

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, u_a = compute_prune(x, self._mode, self._tome_info, self._cache)

            x = u_a(self.attn1(m_a(self.norm1(x), prune=self._tome_info['args']['prune']),
                               context=context if self.disable_self_attn else None)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x

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

            return hidden_states

    return ToMeBlock


def patch_unet(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedUnet(block_class):
        def forward(
                self,
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:

            skip_this_step = self._cache_bus.skip_this_step
            if skip_this_step and self._cache_bus.prev_noise[-1] is not None:
                print("======================================")
                print("\033[96mDEBUG\033[0m: Skip step")
                print("======================================")
                self._cache_bus.last_skip_step = self._cache_bus.step  # todo dis-alignment between bus and cache
                self._cache_bus.skip_this_step = False
                self._cache_bus.skipping_path.append(self._cache_bus.step)

                sample = self._cache_bus.prev_noise[-1].clone()
                for i in range(1): # assume this is second order
                    self._cache_bus.prev_noise[i] = self._cache_bus.prev_noise[i + 1]
                self._cache_bus.prev_noise[-1] = sample
                self._cache_bus.step += 1

                return UNet2DConditionOutput(sample=sample)

            else:
                default_overall_up_factor = 2 ** self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                for dim in sample.shape[-2:]:
                    if dim % default_overall_up_factor != 0:
                        # Forward upsample size to force interpolation output size.
                        forward_upsample_size = True
                        break

                if attention_mask is not None:
                    # assume that mask is expressed as:
                    #   (1 = keep,      0 = discard)
                    # convert mask into a bias that can be added to attention scores:
                    #       (keep = +0,     discard = -10000.0)
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # convert encoder_attention_mask to a bias the same way we do for attention_mask
                if encoder_attention_mask is not None:
                    encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
                    encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                timesteps = timestep
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])

                t_emb = self.time_proj(timesteps)
                t_emb = t_emb.to(dtype=sample.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)
                aug_emb = None

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                elif self.config.addition_embed_type == "text_image":
                    # Kandinsky 2.1 - style
                    if "image_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                        )

                    image_embs = added_cond_kwargs.get("image_embeds")
                    text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
                    aug_emb = self.add_embedding(text_embs, image_embs)
                elif self.config.addition_embed_type == "text_time":
                    # SDXL - style
                    if "text_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                        )
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    if "time_ids" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                        )
                    time_ids = added_cond_kwargs.get("time_ids")
                    time_embeds = self.add_time_proj(time_ids.flatten())
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                    add_embeds = add_embeds.to(emb.dtype)
                    aug_emb = self.add_embedding(add_embeds)
                elif self.config.addition_embed_type == "image":
                    # Kandinsky 2.2 - style
                    if "image_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                        )
                    image_embs = added_cond_kwargs.get("image_embeds")
                    aug_emb = self.add_embedding(image_embs)
                elif self.config.addition_embed_type == "image_hint":
                    # Kandinsky 2.2 - style
                    if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                        )
                    image_embs = added_cond_kwargs.get("image_embeds")
                    hint = added_cond_kwargs.get("hint")
                    aug_emb, hint = self.add_embedding(image_embs, hint)
                    sample = torch.cat([sample, hint], dim=1)

                emb = emb + aug_emb if aug_emb is not None else emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
                elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
                    # Kadinsky 2.1 - style
                    if "image_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                        )

                    image_embeds = added_cond_kwargs.get("image_embeds")
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
                elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
                    # Kandinsky 2.2 - style
                    if "image_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                        )
                    image_embeds = added_cond_kwargs.get("image_embeds")
                    encoder_hidden_states = self.encoder_hid_proj(image_embeds)
                elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
                    if "image_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                        )
                    image_embeds = added_cond_kwargs.get("image_embeds")
                    image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

                # 2. pre-process
                sample = self.conv_in(sample)

                # 2.5 GLIGEN position net
                if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                    cross_attention_kwargs = cross_attention_kwargs.copy()
                    gligen_args = cross_attention_kwargs.pop("gligen")
                    cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

                # 3. down
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
                if USE_PEFT_BACKEND:
                    # weight the lora layers by setting `lora_scale` for each PEFT layer
                    scale_lora_layers(self, lora_scale)

                is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
                # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
                is_adapter = down_intrablock_additional_residuals is not None
                # maintain backward compatibility for legacy usage, where
                #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
                #       but can only use one or the other
                if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                    down_intrablock_additional_residuals = down_block_additional_residuals
                    is_adapter = True

                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        # For t2i-adapter CrossAttnDownBlock2D
                        additional_residuals = {}
                        if is_adapter and len(down_intrablock_additional_residuals) > 0:
                            additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_attention_mask=encoder_attention_mask,
                            **additional_residuals,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                        if is_adapter and len(down_intrablock_additional_residuals) > 0:
                            sample += down_intrablock_additional_residuals.pop(0)

                    down_block_res_samples += res_samples

                if is_controlnet:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                    ):
                        down_block_res_sample = down_block_res_sample + down_block_additional_residual
                        new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                    down_block_res_samples = new_down_block_res_samples

                # 4. mid
                if self.mid_block is not None:
                    if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_attention_mask=encoder_attention_mask,
                        )
                    else:
                        sample = self.mid_block(sample, emb)

                    # To support T2I-Adapter-XL
                    if (
                            is_adapter
                            and len(down_intrablock_additional_residuals) > 0
                            and sample.shape == down_intrablock_additional_residuals[0].shape
                    ):
                        sample += down_intrablock_additional_residuals.pop(0)

                if is_controlnet:
                    sample = sample + mid_block_additional_residual

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                            scale=lora_scale,
                        )

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                # clone the sample to cache for skipping step
                for i in range(1):
                    self._cache_bus.prev_noise[i] = self._cache_bus.prev_noise[i + 1]
                self._cache_bus.prev_noise[-1] = sample
                self._cache_bus.step += 1

                if USE_PEFT_BACKEND:
                    # remove `lora_scale` from each PEFT layer
                    unscale_lora_layers(self, lora_scale)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)

    return PatchedUnet


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
    bus = diffusion_model._cache_bus

    bus.prev_x0 = None
    bus.prev_prev_x0 = None
    bus.temporal_score = None
    bus.prev_sample_list = [None, None]

    bus.prev_momentum = None
    bus.last_skip_step = 0  # align with step in cache bus
    bus.skip_this_step = False
    bus.prev_sample = None
    bus.b_idx = None

    bus.abs_momentum_list = []
    bus.rel_momentum_list = []
    bus.skipping_path = []

    bus.pred_m_m_1 = None
    bus.taylor_m_m_1 = None

    bus.pred_error_list = []
    bus.taylor_error_list = []

    bus.x0_list = []
    bus.xt_list = []

    bus.step = 0
    bus.c_step = 1

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

        test_skip_path: List[int] = None,
        max_fix: int = 5 * 1024,
        threshold_token: float = 0.15,
        threshold_map: float = 0.015,
        cache_interval: int = 3,
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

        f"threshold_token: {threshold_token}\n"
        f"threshold_map: {threshold_map}\n"
        f"max_fix: {max_fix}"
        f"cache_interval: {cache_interval}"
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

            "test_skip_path": test_skip_path,

            "threshold_token": threshold_token,
            "threshold_map": threshold_map,
            "max_fix": max_fix,
            "cache_interval": cache_interval
        }
    }

    hook_tome_model(diffusion_model)

    diffusion_model.__class__ = patch_unet(diffusion_model.__class__)
    diffusion_model._cache_bus = CacheBus()
    diffusion_model._cache_bus._tome_info = diffusion_model._tome_info

    solver.__class__ = patch_solver(solver.__class__)
    solver._cache_bus = diffusion_model._cache_bus

    index = 0
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__, mode=mode)
            module._tome_info = diffusion_model._tome_info
            module._cache = Cache(cache_bus=diffusion_model._cache_bus, index=index)
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
    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in
                          model._bus.model_outputs.items()}
    np.savez(file_name, **numpy_feature_maps)

    numpy_feature_maps = {str(k): [fm.cpu().numpy() if isinstance(fm, torch.Tensor) else fm for fm in v] for k, v in
                          model._bus.model_outputs_change.items()}
    np.savez("outputs/model_outputs_change.npz", **numpy_feature_maps)

