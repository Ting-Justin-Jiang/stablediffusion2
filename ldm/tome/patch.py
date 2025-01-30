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
from diffusers.utils.torch_utils import randn_tensor
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->EulerDiscrete
class EulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class CacheBus:
    """A Bus class for overall control."""

    def __init__(self):
        # == Variable Caching ==
        self.prev_x0 = [None, None]
        self.prev_xt = None
        self.prev_epsilon = [None, None]
        self.prev_f = [None, None, None]

        # == Variable Prediction ==
        self.pred_m_m_1 = None
        self.taylor_m_m_1 = None
        self.pred_epsilon = None

        self.temporal_score = None
        self.step = 0

        # save functions calculated for given step
        self.m_a = None  # todo untested
        self.u_a = None

        # indicator for re-calculation
        self.c_step = 1  # todo untested, just because dpm++ 2M starts from 2
        self.ind_step = None  # todo untested
        self.cons_skip = 0

        # model-level accel
        self.last_skip_step = 0  # align with step in cache bus
        self.skip_this_step = False

        # remove comment to log
        # self.model_outputs = {}
        # self.model_outputs_change = {}
        self.pred_error_list = []
        self.taylor_error_list = []

        self.abs_momentum_list = []
        self.rel_momentum_list = []
        self.skipping_path = []


class Cache:
    def __init__(self, index: int, cache_bus: CacheBus, broadcast_range: int = 1):
        self.cache_bus = cache_bus
        self.feature_map = None
        self.index = index

    # == 1. Cache Merge Operations == #
    def push(self, x: torch.Tensor, index: torch.Tensor = None) -> None:
        if index is None:
            # x would be the entire feature map during the first cache update
            self.feature_map = x
        else:
            # x would be the dst (updated) tokens during subsequent cache updates
            self.feature_map.scatter_(dim=-2, index=index, src=x)

    def pop(self, index: torch.Tensor) -> torch.Tensor:
        # Retrieve the src tokens from the cached feature map
        x = torch.gather(self.feature_map, dim=-2, index=index)
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
    class PatchedEulerDiscreteScheduler(solver_class):
        def step(
                self,
                model_output: torch.FloatTensor,
                timestep: Union[float, torch.FloatTensor],
                sample: torch.FloatTensor,
                s_churn: float = 0.0,
                s_tmin: float = 0.0,
                s_tmax: float = float("inf"),
                s_noise: float = 1.0,
                generator: Optional[torch.Generator] = None,
                return_dict: bool = True,
        ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:

            if self.step_index is None:
                self._init_step_index(timestep)

            sigma = self.sigmas[self.step_index]
            gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device,
                                 generator=generator)

            eps = noise * s_noise
            sigma_hat = sigma * (gamma + 1)

            if gamma > 0:
                sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # == Conduct the step skipping identified from last step == #
            if self._cache_bus.skip_this_step and self._cache_bus.pred_m_m_1 is not None:
                pred_original_sample = self._cache_bus.pred_m_m_1
                self._cache_bus.skip_this_step = False
            else:

                if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
                    pred_original_sample = model_output
                elif self.config.prediction_type == "epsilon":
                    pred_original_sample = sample - sigma_hat * model_output
                elif self.config.prediction_type == "v_prediction":
                    # denoised = model_output * c_out + input * c_skip
                    pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
                else:
                    raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`")

            # == Patch begin
            N = self.timesteps.shape[0]

            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1] / (torch.sqrt(1 + self.sigmas[self.step_index + 1] ** 2)),
                self.sigmas[self.step_index] / (torch.sqrt(1 + self.sigmas[self.step_index] ** 2)),
                self.sigmas[self.step_index - 1] / (torch.sqrt(1 + self.sigmas[self.step_index - 1] ** 2)),
            )

            self.betas = self.betas.to('cuda')

            beta_n = self.betas[int(self.timesteps[self.step_index - 1])]
            beta_n_1 = self.betas[int(self.timesteps[min(N - 1, self.step_index)])]
            beta_n_m1 = self.betas[int(self.timesteps[self.step_index - 2])]

            s_alpha_cumprod_n = torch.sqrt(self.alphas[int(self.timesteps[self.step_index - 1])])
            s_alpha_cumprod_n_1 = torch.sqrt(self.alphas[int(self.timesteps[min(N - 1, self.step_index)])])
            s_alpha_cumprod_n_m1 = torch.sqrt(self.alphas[int(self.timesteps[self.step_index - 2])])

            delta = 1 / N  # step size correlates with number of inference step

            m0 = pred_original_sample / s_alpha_cumprod_n

            if m0.shape[-1] == 128:
                guidance = 5.0
            else: guidance = 7.5

            epsilon_0 = self._cache_bus.prev_epsilon[-1][0] + guidance * (
                    self._cache_bus.prev_epsilon[-1][1] - self._cache_bus.prev_epsilon[-1][0])

            # == Criteria == #
            f = (- 0.5 * beta_n * N * sample) + (0.5 * beta_n * N / sigma_s0) * epsilon_0

            if self._cache_bus.prev_f[0] is not None:
                max_interval = self._cache_bus._tome_info['args']['max_interval']
                acc_range = self._cache_bus._tome_info['args']['acc_range']

                momentum = (self._cache_bus.prev_f[-1] - self._cache_bus.prev_f[-2]) - (self._cache_bus.prev_f[-2] - self._cache_bus.prev_f[-3])
                momentum_mean = momentum.mean()
                self._cache_bus.rel_momentum_list.append((momentum_mean.item()))

                if self._cache_bus.cons_skip >= max_interval:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                elif momentum_mean <= 0 and self._cache_bus.step in range(acc_range[0], acc_range[1]):
                    # == Here we skip step
                    self._cache_bus.skip_this_step = True

                    # Calculate m_m_1
                    m1 = self._cache_bus.prev_x0[-1]
                    epsilon_1 = self._cache_bus.prev_epsilon[-2][0] + guidance * (self._cache_bus.prev_epsilon[-2][1] - self._cache_bus.prev_epsilon[-2][0])

                    # == Calculate the future data prediction in a Adam Moulton fashion ==
                    term_1 = (1 + 0.25 * delta * N * beta_n) * s_alpha_cumprod_n * m0
                    term_2 = 0.25 * delta * N * beta_n_1 * s_alpha_cumprod_n_1 * m1
                    term_3 = ((1 + 0.25 * delta * N * beta_n) * sigma_s0 - sigma_s1 - (delta * N * beta_n) / (4 * sigma_s0)) * epsilon_0
                    term_4 = (0.25 * delta * N * beta_n_1 * sigma_t - (delta * N * beta_n_1) / (4 * sigma_t)) * epsilon_1

                    m_m_1 = (term_1 + term_2 + term_3 + term_4) / s_alpha_cumprod_n_m1
                    self._cache_bus.pred_m_m_1 = m_m_1.clone()

                    self._cache_bus.cons_skip += 1

                else:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                # Here we conduct token-wise pruning:
                if not self._cache_bus.skip_this_step:
                    # == Define the masking on pruning
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    momentum_mean = momentum.mean(dim=1)
                    momentum_flat = momentum_mean.view(-1)
                    mask = momentum_flat >= 0
                    num_mask = mask.sum().item()
                    score = torch.zeros_like(momentum_flat)

                    if num_mask > max_fix:
                        _, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                        score[topk_indices] = 1.0
                    else:
                        score[mask] = 1.0

                    self._cache_bus.temporal_score = score.view_as(momentum_mean)

            for i in range(2):
                self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
            self._cache_bus.prev_f[-1] = f

            for j in range(1):
                self._cache_bus.prev_x0[j] = self._cache_bus.prev_x0[j + 1]
            self._cache_bus.prev_x0[-1] = m0
            self._cache_bus.prev_xt = sample
            # == Patch end ==

            # 2. Convert to an ODE derivative
            derivative = (sample - pred_original_sample) / sigma_hat

            dt = self.sigmas[self.step_index + 1] - sigma_hat

            prev_sample = sample + derivative * dt

            # upon completion increase step index by one
            self._step_index += 1

            if not return_dict:
                return (prev_sample,)

            return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


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

            # == Conduct the step skipping identified from last step == #
            if self._cache_bus.skip_this_step and self._cache_bus.pred_m_m_1 is not None:
                model_output = self._cache_bus.pred_m_m_1
                self._cache_bus.skip_this_step = False
            else:
                model_output = self.convert_model_output(model_output, sample=sample)

            m0 = model_output

            if m0.shape[-1] == 128:
                guidance = 5.0
            else: guidance = 7.5

            N = self.timesteps.shape[0]

            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )

            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

            beta_n = self.betas[self.timesteps[self.step_index - 1]]
            beta_n_1 = self.betas[self.timesteps[min(N - 1, self.step_index)]]
            beta_n_m1 = self.betas[self.timesteps[self.step_index - 2]]

            s_alpha_cumprod_n = self.alpha_t[self.timesteps[self.step_index - 1]]
            s_alpha_cumprod_n_1 = self.alpha_t[self.timesteps[min(N - 1, self.step_index)]]
            s_alpha_cumprod_n_m1 = self.alpha_t[self.timesteps[self.step_index - 2]]

            delta = 1 / N  # step size correlates with number of inference step

            epsilon_0 = self._cache_bus.prev_epsilon[-1][0] + guidance * (self._cache_bus.prev_epsilon[-1][1] - self._cache_bus.prev_epsilon[-1][0])

            # == Criteria == #
            f = (- 0.5 * beta_n * N * sample) + (0.5 * beta_n * N / sigma_s0) * epsilon_0

            if self._cache_bus.prev_f[0] is not None:
                max_interval = self._cache_bus._tome_info['args']['max_interval']
                acc_range = self._cache_bus._tome_info['args']['acc_range']

                momentum = (self._cache_bus.prev_f[-1] - self._cache_bus.prev_f[-2]) - (self._cache_bus.prev_f[-2] - self._cache_bus.prev_f[-3])
                momentum_mean = momentum.mean()
                self._cache_bus.rel_momentum_list.append((momentum_mean.item()))

                if self._cache_bus.cons_skip >= max_interval:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                elif momentum_mean <= 0 and self._cache_bus.step in range(acc_range[0], acc_range[1]):
                    # == Here we skip step
                    self._cache_bus.skip_this_step = True

                    # Calculate m_m_1
                    m1 = self._cache_bus.prev_x0[-1]
                    m2 = self._cache_bus.prev_x0[-2]

                    # CFG to the noise
                    epsilon_1 = self._cache_bus.prev_epsilon[-2][0] + guidance * (
                                self._cache_bus.prev_epsilon[-2][1] - self._cache_bus.prev_epsilon[-2][0])

                    # == Calculate the future data prediction in a Adam Moulton fashion ==
                    term_1 = (1 + 0.25 * delta * N * beta_n) * s_alpha_cumprod_n * m0
                    term_2 = 0.25 * delta * N * beta_n_1 * s_alpha_cumprod_n_1 * m1
                    term_3 = ((1 + 0.25 * delta * N * beta_n) * sigma_s0 - sigma_s1 - (delta * N * beta_n) / (4 * sigma_s0)) * epsilon_0
                    term_4 = (0.25 * delta * N * beta_n_1 * sigma_t - (delta * N * beta_n_1) / (4 * sigma_t)) * epsilon_1

                    m_m_1 = (term_1 + term_2 + term_3 + term_4) / s_alpha_cumprod_n_m1

                    # calculate MSE for our prediction
                    if False:
                        if self._cache_bus.pred_m_m_1 is not None:
                            error = ((self._cache_bus.pred_m_m_1 - m0) ** 2).mean()
                            print(f"\n Our Approximation error - : {error}")
                            self._cache_bus.pred_error_list.append(float(error.item()))
                            print(self._cache_bus.pred_error_list)
                    self._cache_bus.pred_m_m_1 = m_m_1.clone()

                    # calculate MSE for taylor prediction
                    if False:
                        taylor_m_m_1 = m0 + (m0 - m1) + 0.5 * (m0 - 2 * m1 + m2)
                        if self._cache_bus.taylor_m_m_1 is not None:
                            taylor_error = ((self._cache_bus.taylor_m_m_1 - m0) ** 2).mean()
                            print(f"\n Taylor Approximation error - : {taylor_error}")
                            self._cache_bus.taylor_error_list.append(float(taylor_error.item()))
                            print(self._cache_bus.taylor_error_list)
                        self._cache_bus.taylor_m_m_1 = taylor_m_m_1.clone()

                    # # only for testing
                    if self._cache_bus._tome_info['args']['test_skip_path'] and self._cache_bus.step in \
                            self._cache_bus._tome_info['args']['test_skip_path']:
                        self._cache_bus.skip_this_step = True

                    self._cache_bus.cons_skip += 1

                else:
                    self._cache_bus.skip_this_step = False
                    self._cache_bus.cons_skip = 0

                # Here we conduct token-wise pruning:
                if not self._cache_bus.skip_this_step:
                    # # == Calculate the predicted epsilon == #
                    # # Notation: f(t) = -0.5beta(t) g^2(t) = beta(t) beta(t) = N beta_t
                    # assert m_m_1 is not None
                    # assert epsilon_1 is not None
                    #
                    # factor = (1 / (0.25 * delta * N * beta_n / sigma_s0 + sigma_t))
                    # term_1 = - s_alpha_cumprod_n_m1 * m_m_1
                    # term_2 = (1 - 0.5 * delta * (- 0.5 * N * beta_n)) * sample
                    # term_3 = 0.5 * (- 0.5 * delta * N * beta_n_1) * self._cache_bus.prev_xt
                    # term_4 = - (0.25 * delta * N * beta_n_1 / sigma_s1) * epsilon_1
                    #
                    # pred_epsilon = factor * (term_1 + term_2 + term_3 + term_4)
                    # self._cache_bus.pred_epsilon = pred_epsilon

                    # == Define the masking on pruning
                    max_fix = self._cache_bus._tome_info['args']['max_fix']

                    momentum_mean = momentum.mean(dim=1)
                    momentum_flat = momentum_mean.view(-1)
                    mask = momentum_flat >= 0
                    num_mask = mask.sum().item()
                    score = torch.zeros_like(momentum_flat)

                    if num_mask > max_fix:
                        _, topk_indices = torch.topk(momentum_flat, max_fix, largest=True, sorted=False)
                        score[topk_indices] = 1.0
                    else:
                        score[mask] = 1.0

                    self._cache_bus.temporal_score = score.view_as(momentum_mean)

            for i in range(2):
                self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
            self._cache_bus.prev_f[-1] = f

            for j in range(1):
                self._cache_bus.prev_x0[j] = self._cache_bus.prev_x0[j + 1]
            self._cache_bus.prev_x0[-1] = m0
            self._cache_bus.prev_xt = sample

            for k in range(self.config.solver_order - 1):
                self.model_outputs[k] = self.model_outputs[k + 1]

            self.model_outputs[-1] = model_output

            noise = None

            if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
            elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
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

            # # logging 2
            # if score is not None:
            #     score = score.expand([1, 4, 96, 96])
            #     m0 = m0 + score * 2.5
            # feature_map = np.array(m0[0].unsqueeze(dim=0).cpu(), dtype=np.float32).astype(np.float16)
            # self._cache_bus.model_outputs_change[self._cache_bus.c_step] = feature_map

            # self._cache_bus.c_step += 1

            return x_t
    if solver_class.__name__ == "EulerDiscreteScheduler":
        return PatchedEulerDiscreteScheduler
    if solver_class.__name__ == "DPMSolverMultistepScheduler":
        return PatchedDPMSolverMultistepScheduler
    else: raise NotImplementedError



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
            if skip_this_step and self._cache_bus.prev_epsilon[-1] is not None:
                self._cache_bus.last_skip_step = self._cache_bus.step  # todo dis-alignment between bus and cache
                self._cache_bus.skipping_path.append(self._cache_bus.step)

                sample = self._cache_bus.prev_epsilon[-1].clone()
                for i in range(1): # assume this is second order
                    self._cache_bus.prev_epsilon[i] = self._cache_bus.prev_epsilon[i + 1]
                self._cache_bus.prev_epsilon[-1] = sample
                self._cache_bus.step += 1

                return UNet2DConditionOutput(sample=torch.zeros_like(sample))

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
                    self._cache_bus.prev_epsilon[i] = self._cache_bus.prev_epsilon[i + 1]
                self._cache_bus.prev_epsilon[-1] = sample
                self._cache_bus.step += 1

                if USE_PEFT_BACKEND:
                    # remove `lora_scale` from each PEFT layer
                    unscale_lora_layers(self, lora_scale)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)

    return PatchedUnet


def hook_tome_model(model: torch.nn.Module):
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def reset_cache(model: torch.nn.Module):
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    if not is_diffusers:
        diffusion_model = model.model.diffusion_model
    else:
        diffusion_model = model.unet if hasattr(model, "unet") else model

    # reset bus
    bus = diffusion_model._cache_bus

    bus.prev_x0 = [None, None]
    bus.prev_xt = None
    bus.prev_f = [None, None, None]
    bus.temporal_score = None

    bus.prev_momentum = None
    bus.last_skip_step = 0  # align with step in cache bus
    bus.skip_this_step = False
    bus.prev_sample = None

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
    bus.cons_skip = 0


    # re-patch
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
        mode: str = "cache_merge",
        prune: bool = "true",
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,

        acc_range: Tuple[int, int] = (7, 45),
        push_unmerged: bool = True,

        test_skip_path: List[int] = None,
        max_fix: int = 5 * 1024,
        max_interval: int = 3,
):
    # == merging preparation ==
    global DEBUG_MODE
    if DEBUG_MODE: print('Start with \033[95mDEBUG\033[0m mode')
    print('\033[94mApplying Token Merging\033[0m')

    acc_start = max(acc_range[0], 3)

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
        f"acc_range: {acc_start, acc_range[-1]}\n"
        f"push_unmerged: {push_unmerged}\n"

        f"max_fix: {max_fix}"
        f"max_interval: {max_interval}"
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

            "acc_range": (acc_start, acc_range[-1]),
            "push_unmerged": push_unmerged,

            "max_fix": max_fix,
            "max_interval": max_interval,
            "test_skip_path": test_skip_path,
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
            index += 1

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

    print(f"Applied merging patch for BasicTransformerBlock")
    return model


def remove_patch(model: torch.nn.Module):
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

