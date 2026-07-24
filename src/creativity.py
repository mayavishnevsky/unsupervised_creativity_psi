"""Differentiable latent IEM reward for Psi-Sampler.

The notation follows ``Unsupervised Creative Generation``. Equation 14 uses
the frozen FLUX denoiser to predict a clean endpoint from a noisy probe.
Equation 16 collects the weighted prediction errors into ``Phi(x_0)``, and
equation 21 scores a candidate against a fixed cloud of reference features.

Reference endpoints and statistics are computed once without gradients.
Candidate features intentionally keep gradients through every denoiser probe.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.reward_model import register_reward_model
from src.utils import ignore_kwargs, synchronized_time


Tensor = torch.Tensor


@dataclass(frozen=True)
class PromptRecord:
    """A prompt and its stable row id across all configured source files."""

    prompt_id: int
    text: str


class BalancedPromptSampler:
    """Draw deterministic, balanced prompt samples from configurable files."""

    def __init__(self, paths: Sequence[str | Path]):
        if not paths:
            raise ValueError("at least one reference prompt source is required")

        self.paths = tuple(Path(path).expanduser().resolve() for path in paths)
        self._sources: list[list[PromptRecord]] = []
        record_count = 0
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"reference prompt source does not exist: {path}")
            lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"reference prompt source is empty: {path}")

            source = []
            for text in lines:
                source.append(PromptRecord(record_count, text))
                record_count += 1
            self._sources.append(source)

    def sample(
        self,
        count: int,
        seed: int,
        excluded_prompts: Iterable[str] = (),
    ) -> list[PromptRecord]:
        """Sample without replacement within each source after exclusions."""

        count = int(count)
        if count < 1:
            raise ValueError(f"reference sample count must be positive, got {count}")

        excluded = {str(prompt).strip() for prompt in excluded_prompts}
        generator = random.Random(int(seed))
        source_order = list(range(len(self._sources)))
        generator.shuffle(source_order)
        quotas = [count // len(self._sources)] * len(self._sources)
        for source_index in source_order[: count % len(self._sources)]:
            quotas[source_index] += 1

        selected = []
        for source_index, (source, quota) in enumerate(
            zip(self._sources, quotas, strict=True)
        ):
            available = [record for record in source if record.text not in excluded]
            if quota > len(available):
                raise ValueError(
                    f"reference source {self.paths[source_index]} has "
                    f"{len(available)} eligible rows, but its balanced quota is {quota}"
                )
            selected.extend(generator.sample(available, quota))

        generator.shuffle(selected)
        return selected


def iem_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_steps: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return descending log-uniform VE sigma boundaries."""

    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    num_steps = int(num_steps)
    if not (math.isfinite(sigma_min) and math.isfinite(sigma_max)):
        raise ValueError("IEM sigma bounds must be finite")
    if not 0 < sigma_min < sigma_max:
        raise ValueError(
            f"expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}"
        )
    if num_steps < 1:
        raise ValueError(f"IEM needs at least one integration interval, got {num_steps}")

    return torch.linspace(
        math.log(sigma_max),
        math.log(sigma_min),
        num_steps + 1,
        device=device,
        dtype=torch.float32,
    ).exp()


def iem_schedule_terms(
    sigma_schedule: Tensor | Sequence[float],
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return probe sigmas and equation-16 ``delta_gamma`` widths."""

    schedule = torch.as_tensor(sigma_schedule, device=device, dtype=torch.float32)
    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError(
            "sigma_schedule must be one-dimensional with at least two boundaries"
        )
    if not torch.isfinite(schedule).all() or not (schedule > 0).all():
        raise ValueError("sigma_schedule must contain finite positive values")

    gamma_boundaries = schedule.reciprocal().square()
    delta_gamma = gamma_boundaries[1:] - gamma_boundaries[:-1]
    if not (delta_gamma > 0).all():
        raise ValueError("sigma_schedule must descend strictly so gamma ascends")
    return schedule[:-1], delta_gamma


def sample_iem_noise_table(
    sigma_schedule: Tensor | Sequence[float],
    signal_shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Draw the fixed Gaussian probe shared by all references and candidates."""

    probe_sigmas, _ = iem_schedule_terms(sigma_schedule, device=device)
    shape = tuple(int(size) for size in signal_shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"signal_shape must be positive, got {shape}")

    generator = torch.Generator(device=probe_sigmas.device).manual_seed(int(seed))
    return torch.randn(
        (probe_sigmas.numel(), 1, *shape),
        device=probe_sigmas.device,
        dtype=dtype,
        generator=generator,
    )


def expected_psi_reward_rounds(
    num_mcmc_steps: int,
    burn_in: int,
    num_inference_steps: int,
) -> int:
    """Return the pCNL plus SMC reward rounds used for one Psi candidate."""

    num_mcmc_steps = int(num_mcmc_steps)
    burn_in = int(burn_in)
    num_inference_steps = int(num_inference_steps)
    if num_mcmc_steps < 1:
        raise ValueError("num_mcmc_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")

    # pCNL: initial + loop + final. SMC: initial + one per inference step.
    return num_mcmc_steps + burn_in + num_inference_steps + 3


def staged_noise_table_index(
    reward_round_index: int,
    total_reward_rounds: int,
    noise_table_count: int,
) -> int:
    """Divide ordered reward rounds as evenly as possible across noise tables."""

    reward_round_index = int(reward_round_index)
    total_reward_rounds = int(total_reward_rounds)
    noise_table_count = int(noise_table_count)
    if total_reward_rounds < 1:
        raise ValueError("total_reward_rounds must be positive")
    if not 1 <= noise_table_count <= total_reward_rounds:
        raise ValueError(
            "noise_table_count must be between one and total_reward_rounds"
        )
    if not 0 <= reward_round_index < total_reward_rounds:
        raise ValueError("reward_round_index is outside the configured run")

    return reward_round_index * noise_table_count // total_reward_rounds


def repeat_endpoint_conditions(
    prompt_embeds: Tensor,
    pooled_prompt_embeds: Tensor,
    flattened_batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Repeat conditions in the level-major order produced by ``iem_features``."""

    if prompt_embeds.shape[0] < 1:
        raise ValueError("prompt condition batch must be nonempty")
    if prompt_embeds.shape[0] != pooled_prompt_embeds.shape[0]:
        raise ValueError("prompt condition batches must have the same size")

    endpoint_count = prompt_embeds.shape[0]
    flattened_batch_size = int(flattened_batch_size)
    if (
        flattened_batch_size < endpoint_count
        or flattened_batch_size % endpoint_count
    ):
        raise ValueError(
            "flattened probe count must be a positive multiple of endpoint count"
        )

    repeats = flattened_batch_size // endpoint_count
    return (
        prompt_embeds.repeat((repeats,) + (1,) * (prompt_embeds.ndim - 1)),
        pooled_prompt_embeds.repeat(
            (repeats,) + (1,) * (pooled_prompt_embeds.ndim - 1)
        ),
    )


def iem_features(
    x_0: Tensor,
    velocity_prediction: Callable[[Tensor, Tensor], Tensor],
    sigma_schedule: Tensor | Sequence[float],
    noise_table: Tensor,
    level_batch_size: int = 1,
    checkpoint_velocity_prediction: bool = False,
) -> Tensor:
    """Compute the finite IEM feature map ``Phi(x_0)`` from equations 14/16.

    FLUX uses the linear path ``x_t = (1-t)x_0 + t epsilon`` and predicts
    velocity ``v = epsilon - x_0``. Therefore equation 14's clean prediction is
    ``f(x_t) = x_t - t v(x_t, t)``. The denoiser input keeps the endpoint dtype;
    residuals and integration weights are evaluated in float32.
    """

    x_0 = torch.as_tensor(x_0)
    if x_0.ndim < 2 or x_0.shape[0] < 1:
        raise ValueError(
            f"x_0 must have nonempty (batch, ...) shape, got {tuple(x_0.shape)}"
        )

    probe_sigmas, delta_gamma = iem_schedule_terms(
        sigma_schedule, device=x_0.device
    )
    interval_count = probe_sigmas.numel()
    level_batch_size = int(level_batch_size)
    if level_batch_size < 1:
        raise ValueError(
            f"level_batch_size must be positive, got {level_batch_size}"
        )

    noise_table = torch.as_tensor(noise_table)
    expected_tail = tuple(x_0.shape[1:])
    if (
        noise_table.ndim != x_0.ndim + 1
        or noise_table.shape[0] != interval_count
        or noise_table.shape[1] not in (1, x_0.shape[0])
        or tuple(noise_table.shape[2:]) != expected_tail
    ):
        raise ValueError(
            "noise_table must have shape (levels, 1|batch, *signal_shape); "
            f"got {tuple(noise_table.shape)}"
        )

    feature_blocks = []
    broadcast_tail = (1,) * (x_0.ndim - 1)
    for start in range(0, interval_count, level_batch_size):
        stop = min(start + level_batch_size, interval_count)
        sigma = probe_sigmas[start:stop]
        flow_time = sigma / (1.0 + sigma)
        noise = noise_table[start:stop].to(
            device=x_0.device, dtype=torch.float32
        )
        x_0_32 = x_0.float()
        x_t_32 = (
            x_0_32.unsqueeze(0)
            + sigma.reshape((-1, 1) + broadcast_tail) * noise
        ) / (1.0 + sigma).reshape((-1, 1) + broadcast_tail)

        flat_x_t = x_t_32.to(dtype=x_0.dtype).flatten(0, 1)
        flat_t = flow_time[:, None].expand(-1, x_0.shape[0]).reshape(-1)
        if checkpoint_velocity_prediction and torch.is_grad_enabled():
            velocity = checkpoint(
                velocity_prediction,
                flat_x_t,
                flat_t,
                use_reentrant=False,
            )
        else:
            velocity = velocity_prediction(flat_x_t, flat_t)
        velocity = velocity.reshape_as(x_t_32)

        # Equation 14, followed by the weighted residual blocks in equation 16.
        flow_time_32 = flow_time.reshape((-1, 1) + broadcast_tail)
        f_x_t = x_t_32 - flow_time_32 * velocity.float()
        e_gamma = x_0_32.unsqueeze(0) - f_x_t
        weights = (delta_gamma[start:stop] / interval_count).sqrt()
        weighted = weights.reshape((-1, 1) + broadcast_tail) * e_gamma
        feature_blocks.append(weighted.transpose(0, 1).flatten(start_dim=1))

    return torch.cat(feature_blocks, dim=1)


def update_reference_sums(
    count: int,
    sum_phi: Tensor | None,
    sum_phi_squared_norm: Tensor | None,
    features: Tensor,
) -> tuple[int, Tensor, Tensor]:
    """Accumulate ``M``, ``sum Phi(x'_j)``, and ``sum ||Phi(x'_j)||^2``."""

    features = torch.as_tensor(features).float()
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(
            f"features must have nonempty shape (N, D), got {tuple(features.shape)}"
        )

    features_64 = features.to(dtype=torch.float64)
    batch_sum_phi = features_64.sum(dim=0)
    batch_sum_phi_squared_norm = features_64.square().sum()
    count = int(count)

    if count == 0:
        if sum_phi is not None or sum_phi_squared_norm is not None:
            raise ValueError("zero reference count must have empty sums")
        return features.shape[0], batch_sum_phi, batch_sum_phi_squared_norm
    if count < 0 or sum_phi is None or sum_phi_squared_norm is None:
        raise ValueError("positive reference count requires both accumulated sums")

    sum_phi = torch.as_tensor(sum_phi, device=features.device, dtype=torch.float64)
    sum_phi_squared_norm = torch.as_tensor(
        sum_phi_squared_norm, device=features.device, dtype=torch.float64
    )
    if sum_phi.shape != features.shape[1:]:
        raise ValueError("accumulated feature sum has the wrong shape")
    if sum_phi_squared_norm.numel() != 1:
        raise ValueError("accumulated squared-norm sum must be scalar")

    return (
        count + features.shape[0],
        sum_phi + batch_sum_phi,
        sum_phi_squared_norm + batch_sum_phi_squared_norm,
    )


def reference_statistics(
    count: int,
    sum_phi: Tensor,
    sum_phi_squared_norm: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``mu_omega`` and scalar ``v_omega`` exactly as in equation 21."""

    count = int(count)
    if count < 1:
        raise ValueError("reference count must be positive")
    sum_phi = torch.as_tensor(sum_phi, dtype=torch.float64)
    sum_phi_squared_norm = torch.as_tensor(
        sum_phi_squared_norm, device=sum_phi.device, dtype=torch.float64
    )
    if sum_phi_squared_norm.numel() != 1:
        raise ValueError("sum_phi_squared_norm must be scalar")

    mu_omega_64 = sum_phi / count
    v_omega = (
        sum_phi_squared_norm / count - mu_omega_64.square().sum()
    ).clamp_min(0)
    return mu_omega_64.float(), v_omega


def iem_reward_from_reference_statistics(
    features: Tensor,
    mu_omega: Tensor,
    v_omega: Tensor | float,
) -> Tensor:
    """Evaluate equation 21: ``||Phi(x)-mu_omega||^2 + v_omega``."""

    features = torch.as_tensor(features).float()
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {features.shape}")
    mu_omega = torch.as_tensor(
        mu_omega, device=features.device, dtype=torch.float32
    )
    if mu_omega.shape != features.shape[1:]:
        raise ValueError("mu_omega and candidate feature dimensions do not match")
    v_omega = torch.as_tensor(
        v_omega, device=features.device, dtype=torch.float64
    )
    if v_omega.numel() != 1 or not torch.isfinite(v_omega) or v_omega < 0:
        raise ValueError("v_omega must be a finite non-negative scalar")

    return (
        (features - mu_omega).square().sum(dim=1, dtype=torch.float64) + v_omega
    )


@register_reward_model(name="creativity")
class IEMReward(nn.Module):
    """Equation-21 reward using fixed endpoints and table-matched statistics."""

    @ignore_kwargs
    @dataclass
    class Config:
        decode_to_unnormalized: bool = False
        reward_input_type: str = "latent"
        grad_norm: float | None = None
        grad_const_scale: float | None = None
        seed: int = 0
        height: int = 512
        width: int = 512
        sigma_min: float = 1.0
        sigma_max: float = 1000.0
        num_steps: int = 64
        level_batch_size: int = 4
        checkpoint_candidate_features: bool = True
        noise_table_mode: str = "fixed"
        noise_table_count: int = 1
        reference_prompt_files: tuple[str, ...] = ()
        reference_sample_count: int = 48
        reference_batch_size: int = 1
        reference_num_inference_steps: int = 4
        reference_seed: int = 30_000_000
        noise_seed: int = 90_000_000
        num_mcmc_steps: int = 25
        burn_in: int = 25
        num_inference_steps: int = 25
        log_timing: bool = False

    def __init__(self, dtype, device, save_dir, CFG):
        super().__init__()
        del dtype, save_dir
        self.cfg = self.Config(**CFG)
        self.device = torch.device(device)
        self._validate_config()
        self.prompt_sampler = BalancedPromptSampler(
            self.cfg.reference_prompt_files
        )
        self.register_buffer(
            "sigma_schedule",
            iem_sigma_schedule(
                self.cfg.sigma_min,
                self.cfg.sigma_max,
                self.cfg.num_steps,
                device=self.device,
            ),
            persistent=False,
        )
        self.register_buffer("noise_tables", None, persistent=False)
        self.register_buffer("mu_omegas", None, persistent=False)
        self.register_buffer("v_omegas", None, persistent=False)
        self.reference_count = 0
        self.prepared = False
        self.total_reward_rounds = expected_psi_reward_rounds(
            self.cfg.num_mcmc_steps,
            self.cfg.burn_in,
            self.cfg.num_inference_steps,
        )
        self.reward_round_count = 0
        self.reward_minibatch_in_round = 0
        self.active_noise_table_index: int | None = None

    def _validate_config(self) -> None:
        for name in (
            "level_batch_size",
            "noise_table_count",
            "reference_sample_count",
            "reference_batch_size",
            "reference_num_inference_steps",
            "height",
            "width",
        ):
            if int(getattr(self.cfg, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.cfg.reward_input_type != "latent":
            raise ValueError("the creativity reward requires reward_input_type=latent")
        if self.cfg.noise_table_mode not in ("fixed", "staged"):
            raise ValueError("noise_table_mode must be 'fixed' or 'staged'")
        if self.cfg.noise_table_mode == "fixed" and self.cfg.noise_table_count != 1:
            raise ValueError("fixed noise-table mode requires noise_table_count=1")
        total_reward_rounds = expected_psi_reward_rounds(
            self.cfg.num_mcmc_steps,
            self.cfg.burn_in,
            self.cfg.num_inference_steps,
        )
        if (
            self.cfg.noise_table_mode == "staged"
            and not 2 <= self.cfg.noise_table_count <= total_reward_rounds
        ):
            raise ValueError(
                "staged noise_table_count must be between 2 and the total "
                f"reward rounds ({total_reward_rounds})"
            )

    def _timing_start(self):
        if not self.cfg.log_timing:
            return None
        return synchronized_time(self.device)

    def _timing_elapsed(self, start):
        if start is None:
            return 0.0
        return synchronized_time(self.device) - start

    def register_data(self, data) -> None:
        """Reset the noise-table schedule for the next candidate prompt."""

        del data
        self.reward_round_count = 0
        self.reward_minibatch_in_round = 0
        self.active_noise_table_index = None

    def begin_reward_evaluation(self) -> None:
        """Select one table for a complete reward round before minibatching."""

        if (
            self.cfg.noise_table_mode == "staged"
            and self.reward_round_count >= self.total_reward_rounds
        ):
            raise RuntimeError(
                "Psi requested more reward rounds than configured; update "
                "num_mcmc_steps, burn_in, or num_inference_steps"
            )

        if self.cfg.noise_table_mode == "fixed":
            table_index = 0
        else:
            table_index = staged_noise_table_index(
                self.reward_round_count,
                self.total_reward_rounds,
                self.cfg.noise_table_count,
            )

        self.active_noise_table_index = table_index
        self.reward_round_count += 1
        self.reward_minibatch_in_round = 0

    def _latent_image_ids(self, pipe, batch_size: int, dtype: torch.dtype) -> Tensor:
        latent_height = int(self.cfg.height) // (pipe.pipe.vae_scale_factor * 2)
        latent_width = int(self.cfg.width) // (pipe.pipe.vae_scale_factor * 2)
        return pipe.pipe._prepare_latent_image_ids(
            batch_size,
            latent_height,
            latent_width,
            self.device,
            dtype,
        )

    @torch.no_grad()
    def prepare_references(
        self,
        pipe,
        excluded_prompts: Sequence[str],
    ) -> None:
        """Generate the fixed reference endpoints and equation-21 statistics."""

        if self.prepared:
            raise RuntimeError("IEM references have already been prepared")

        total_start = synchronized_time(self.device)
        pipe.pipe.transformer.eval().requires_grad_(False)
        sampling_start = self._timing_start()
        records = self.prompt_sampler.sample(
            self.cfg.reference_sample_count,
            seed=int(self.cfg.reference_seed) + int(self.cfg.seed),
            excluded_prompts=excluded_prompts,
        )
        reference_prompt_sampling_seconds = self._timing_elapsed(sampling_start)
        reference_encoding_seconds = 0.0
        reference_generation_seconds = 0.0
        reference_feature_seconds = 0.0
        reference_statistics_seconds = 0.0
        table_count = int(self.cfg.noise_table_count)
        counts = [0] * table_count
        sum_phis = [None] * table_count
        sum_phi_squared_norms = [None] * table_count
        batch_size = int(self.cfg.reference_batch_size)

        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            prompts = [record.text for record in batch_records]

            stage_start = self._timing_start()
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                device=self.device,
            )
            reference_encoding_seconds += self._timing_elapsed(stage_start)
            generators = [
                torch.Generator(device=self.device).manual_seed(
                    int(self.cfg.reference_seed)
                    + int(self.cfg.seed)
                    + record.prompt_id
                )
                for record in batch_records
            ]

            stage_start = self._timing_start()
            endpoints = pipe.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=pipe.cfg.guidance_scale,
                height=int(self.cfg.height),
                width=int(self.cfg.width),
                num_inference_steps=int(self.cfg.reference_num_inference_steps),
                num_images_per_prompt=1,
                generator=generators,
                output_type="latent",
            ).images
            reference_generation_seconds += self._timing_elapsed(stage_start)

            if self.noise_tables is None:
                self.noise_tables = torch.stack(
                    [
                        sample_iem_noise_table(
                            self.sigma_schedule,
                            endpoints.shape[1:],
                            device=self.device,
                            dtype=torch.float32,
                            seed=(
                                int(self.cfg.noise_seed)
                                + int(self.cfg.seed)
                                + table_index
                            ),
                        )
                        for table_index in range(table_count)
                    ],
                    dim=0,
                )
            elif tuple(endpoints.shape[1:]) != tuple(self.noise_tables.shape[3:]):
                raise ValueError("reference endpoint shape changed during preparation")

            latent_image_ids = self._latent_image_ids(
                pipe, len(batch_records), prompt_embeds.dtype
            )

            def predict_reference(x_t: Tensor, flow_time: Tensor) -> Tensor:
                condition, pooled_condition = repeat_endpoint_conditions(
                    prompt_embeds, pooled_prompt_embeds, x_t.shape[0]
                )
                return pipe.predict_conditioned(
                    x_t,
                    flow_time,
                    prompt_embeds=condition,
                    pooled_prompt_embeds=pooled_condition,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                )

            for table_index in range(table_count):
                stage_start = self._timing_start()
                features = iem_features(
                    endpoints,
                    predict_reference,
                    self.sigma_schedule,
                    self.noise_tables[table_index],
                    level_batch_size=int(self.cfg.level_batch_size),
                )
                reference_feature_seconds += self._timing_elapsed(stage_start)

                stage_start = self._timing_start()
                (
                    counts[table_index],
                    sum_phis[table_index],
                    sum_phi_squared_norms[table_index],
                ) = update_reference_sums(
                    counts[table_index],
                    sum_phis[table_index],
                    sum_phi_squared_norms[table_index],
                    features,
                )
                reference_statistics_seconds += self._timing_elapsed(stage_start)

        stage_start = self._timing_start()
        statistics = [
            reference_statistics(count, sum_phi, sum_phi_squared_norm)
            for count, sum_phi, sum_phi_squared_norm in zip(
                counts,
                sum_phis,
                sum_phi_squared_norms,
                strict=True,
            )
        ]
        self.mu_omegas = torch.stack([mu_omega for mu_omega, _ in statistics])
        self.v_omegas = torch.stack([v_omega for _, v_omega in statistics])
        reference_statistics_seconds += self._timing_elapsed(stage_start)
        if len(set(counts)) != 1:
            raise RuntimeError("noise tables produced inconsistent reference counts")
        self.reference_count = counts[0]
        self.prepared = True
        elapsed = synchronized_time(self.device) - total_start
        print(
            f"Prepared {self.reference_count} fixed IEM references in {elapsed:.1f}s "
            f"using {self.cfg.num_steps} probe levels and {table_count} noise tables",
            flush=True,
        )
        if self.cfg.log_timing:
            print(
                "[timing] phase=reference_preparation "
                f"prompt_sampling_seconds={reference_prompt_sampling_seconds:.3f} "
                f"prompt_encoding_seconds={reference_encoding_seconds:.3f} "
                f"endpoint_generation_seconds={reference_generation_seconds:.3f} "
                f"iem_features_seconds={reference_feature_seconds:.3f} "
                f"statistics_seconds={reference_statistics_seconds:.3f} "
                f"noise_table_count={table_count} "
                f"total_seconds={elapsed:.3f}",
                flush=True,
            )

    def forward(self, x_0: Tensor, pipe) -> Tensor:
        """Score candidate Tweedie endpoints while retaining denoiser gradients."""

        if not self.prepared:
            raise RuntimeError("prepare_references must run before IEM scoring")
        if self.active_noise_table_index is None:
            raise RuntimeError(
                "begin_reward_evaluation must run before IEM scoring"
            )

        table_index = self.active_noise_table_index
        noise_table = self.noise_tables[table_index]
        mu_omega = self.mu_omegas[table_index]
        v_omega = self.v_omegas[table_index]
        if tuple(x_0.shape[1:]) != tuple(noise_table.shape[2:]):
            raise ValueError(
                "candidate latent shape does not match the active IEM noise table"
            )

        self.reward_minibatch_in_round += 1
        feature_start = self._timing_start()
        features = iem_features(
            x_0,
            pipe.predict,
            self.sigma_schedule,
            noise_table,
            level_batch_size=int(self.cfg.level_batch_size),
            checkpoint_velocity_prediction=bool(
                self.cfg.checkpoint_candidate_features
            ),
        )
        feature_seconds = self._timing_elapsed(feature_start)

        score_start = self._timing_start()
        score = iem_reward_from_reference_statistics(
            features, mu_omega, v_omega
        ).float()
        score_seconds = self._timing_elapsed(score_start)
        if self.cfg.log_timing:
            print(
                "[timing] phase=iem_candidate "
                f"reward_round={self.reward_round_count} "
                f"minibatch_in_round={self.reward_minibatch_in_round} "
                f"noise_table={table_index} "
                f"batch_size={x_0.shape[0]} "
                f"features_seconds={feature_seconds:.3f} "
                f"equation21_seconds={score_seconds:.3f}",
                flush=True,
            )
        return score
