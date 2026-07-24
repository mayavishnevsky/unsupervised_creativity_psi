# IEM in Psi-Sampler

This document describes the Information Estimation Metric (IEM) reward in this
Psi-Sampler repository. The mathematical source is equations 14, 16, and 21 in
`../unsupervised_creativity_ram/Unsupervised_Creative_Generation.pdf`.

The implementation is an inference-time reward. It does not train FLUX model
parameters. It differentiates the reward with respect to the current Psi
particle so pCNL and SMC can move that particle toward a larger IEM score.

## Implementation map

| File | Responsibility |
| --- | --- |
| [`src/creativity.py`](src/creativity.py#L1) | Prompt sampling, sigma schedule, fixed or staged noise tables, equations 14/16/21, reference preparation, and the registered `creativity` reward. |
| [`src/flux_pipeline.py`](src/flux_pipeline.py#L543) | Produces the outer Tweedie candidate, routes latent rewards without decoding first, computes reward gradients, and supplies native conditioned FLUX velocity predictions. |
| [`main.py`](main.py#L45) | Constructs the reward, prepares references once, then runs pCNL and SMC for each candidate prompt. |
| [`config/creativity.yaml`](config/creativity.yaml#L1) | Full/default IEM, pCNL, and SMC parameters. |
| [`experiment_scripts/run_creativity_smoke_h200.slurm`](experiment_scripts/run_creativity_smoke_h200.slurm#L62) | Current reduced H200 integration run with timing and matched baseline output. |
| [`tests/test_creativity.py`](tests/test_creativity.py#L1) | Formula, determinism, gradient, conditioning, and existing-reward regression tests. |

## End-to-end flow

For one program invocation, the flow is:

1. Load FLUX.1-schnell in bfloat16, compile the transformer and VAE, and freeze
   their parameters. The model choice and freezing are in
   [`src/flux_pipeline.py:75`](src/flux_pipeline.py#L75).
2. Read all candidate prompts from the selected dataset.
3. Before processing any candidate, generate one fixed set of reference
   endpoints. Every candidate prompt in the dataset is excluded from reference
   sampling at [`main.py:59`](main.py#L59).
4. Compute equation-21 reference statistics for each configured noise table.
5. For each candidate prompt, encode its conditions and sample raw FLUX noise.
6. Run iterative pCNL MCMC to obtain reward-aware initial particles.
7. Run iterative SMC to denoise and continue reward-guided particle updates.
8. Decode and save the selected Psi image.
9. When baseline comparison is enabled, run vanilla FLUX from the preserved
   pre-MCMC noise and save the matched baseline image.

Reference preparation is outside the candidate loop
([`main.py:59`](main.py#L59)); pCNL and SMC are inside it
([`main.py:98`](main.py#L98), [`main.py:118`](main.py#L118)).

## Reference prompts

### Sources and counts

The full/default configuration uses three files
([`config/creativity.yaml:85`](config/creativity.yaml#L85)):

| Source | Rows in the current checkout | Default references |
| --- | ---: | ---: |
| PickScore | 25,432 | 16 |
| OCR | 19,653 | 16 |
| GenEval | 50,000 | 16 |
| **Total** | **95,085** | **48** |

`reference_sample_count=48` and the three-way balance are configured at
[`config/creativity.yaml:89`](config/creativity.yaml#L89). The balanced sampler
computes a per-file quota at
[`src/creativity.py:73`](src/creativity.py#L73). If the requested total is not
divisible by the number of files, the remainder is assigned to a
deterministically shuffled subset of sources.

Sampling is without replacement within each file. Exclusion is exact after
whitespace stripping: every row whose text equals any candidate prompt is
removed before sampling
([`src/creativity.py:73`](src/creativity.py#L73),
[`src/creativity.py:85`](src/creativity.py#L85)).

Consequences:

- A candidate prompt cannot also be a reference in the same invocation.
- References are not removed or refreshed per candidate.
- There is no reservoir, priority sampling, or epoch-dependent reference bank.
- The same 48 reference endpoints are used by every candidate. There is one
  matched `mu_omega, v_omega` pair per configured noise table.
- References are not persisted to disk. A new invocation regenerates them.

### Determinism

Every prompt row receives a stable `prompt_id` while the source files are read
([`src/creativity.py:57`](src/creativity.py#L57)).

The selected prompt set uses:

```text
reference_seed + seed
```

at [`src/creativity.py:575`](src/creativity.py#L575). Each reference endpoint
uses:

```text
reference_seed + seed + prompt_id
```

at [`src/creativity.py:603`](src/creativity.py#L603). With unchanged source
files, source order, exclusions, and configuration, reference selection and
initial endpoint noise are reproducible.

### Reference endpoint generation

For each selected prompt:

- The prompt is encoded independently
  ([`src/creativity.py:596`](src/creativity.py#L596)).
- Vanilla FLUX.1-schnell generates one packed latent endpoint
  ([`src/creativity.py:612`](src/creativity.py#L612)).
- The default uses four denoising steps and one image per prompt
  ([`config/creativity.yaml:91`](config/creativity.yaml#L91)).
- Reference generation and feature calculation are under
  `@torch.no_grad()` ([`src/creativity.py:561`](src/creativity.py#L561)).
- The transformer is explicitly put in evaluation mode and frozen
  ([`src/creativity.py:573`](src/creativity.py#L573)).

With the default batch size of one, endpoint generation therefore performs
`48 * 4 = 192` native FLUX denoising steps during one-time startup.

## IEM schedule and shared noise

The default schedule has:

```text
sigma_min = 1
sigma_max = 1000
num_steps = G = 64
```

See [`config/creativity.yaml:77`](config/creativity.yaml#L77). The code creates
`G + 1 = 65` descending log-uniform sigma boundaries
([`src/creativity.py:97`](src/creativity.py#L97)). For interval `i`:

```math
\gamma_i = \frac{1}{\sigma_i^2},
\qquad
\Delta\gamma_i = \gamma_{i+1} - \gamma_i > 0.
```

This is implemented at
[`src/creativity.py:126`](src/creativity.py#L126).

One noise tensor is sampled per integration level. Each table has shape:

```text
(num_steps, 1, *latent_shape)
```

The singleton endpoint dimension is intentional. At a given level and active
table, every reference endpoint and every candidate in the reward round uses
the same Gaussian noise. Different levels and different tables use different
noise.

Tables are created once from the first reference endpoint shape. Table `k`
uses:

```text
noise_seed + seed + k
```

The storage shape is:

```text
(noise_table_count, num_steps, 1, *latent_shape)
```

Table construction is at
[`src/creativity.py:625`](src/creativity.py#L625).

For every table, the implementation computes a separate equation-21
`mu_omega, v_omega` pair from the same fixed reference endpoints. A candidate
must always use the statistics produced with its active noise table; mixing a
table with statistics from another table would no longer be the intended
shared-noise comparison.

Two modes are available:

- `fixed`: the default. `noise_table_count` must be 1, and all reward rounds
  use table 0. This preserves the original Psi IEM behavior.
- `staged`: reward rounds advance through `noise_table_count` contiguous,
  nearly equal windows. With default sampler lengths and eight tables, the 78
  rounds are assigned `[10, 10, 10, 9, 10, 10, 10, 9]` rounds per table.

Round counting and assignment are implemented at
[`src/creativity.py:171`](src/creativity.py#L171) and
[`src/creativity.py:192`](src/creativity.py#L192).

The table advances once per **reward round**, before
`grad_minibatch_size` splits the chains or particles. Therefore the default
520 reward-model minibatch calls do not produce 520 table changes. All five
pCNL chains or all ten SMC particles in one round use the same table.

`level_batch_size` controls how many levels are assembled together in
`iem_features`; it does not create tables or change the selected noise.

## Equation 14: clean prediction

IEM starts from an endpoint latent `x_0`. At VE noise level `sigma`, the code
constructs:

```math
x_t = \frac{x_0 + \sigma\epsilon}{1+\sigma},
\qquad
t = \frac{\sigma}{1+\sigma}.
```

See [`src/creativity.py:299`](src/creativity.py#L299).

This is the native FLUX linear path:

```math
x_t = (1-t)x_0 + t\epsilon.
```

FLUX predicts linear rectified-flow velocity:

```math
v(x_t,t) = \epsilon - x_0.
```

Therefore equation 14's clean prediction is:

```math
f(x_t,t) = x_t - t\,v(x_t,t).
```

No extra `t` factor is missing. Substituting the exact velocity gives:

```math
x_t - t(\epsilon-x_0)
= (1-t)x_0+t\epsilon-t\epsilon+tx_0
= x_0.
```

The implementation predicts velocity and computes `f` at
[`src/creativity.py:306`](src/creativity.py#L306) and
[`src/creativity.py:310`](src/creativity.py#L310).

### Native velocity versus scheduler conversion

There are two denoiser roles:

1. **Outer Psi denoiser.** It predicts the current particle's velocity and
   constructs the Tweedie endpoint. The default outer sampler converts between
   the configured VP schedule and FLUX's native linear schedule
   ([`src/flux_pipeline.py:360`](src/flux_pipeline.py#L360)). The generalized
   Tweedie formula is at
   [`src/flux_pipeline.py:401`](src/flux_pipeline.py#L401).
2. **IEM probe denoiser.** It must use native FLUX linear time because equation
   14 uses `f=x_t-t*v`. Candidate probes call `pipe.predict` directly
   ([`src/creativity.py:744`](src/creativity.py#L744)); reference probes call
   `predict_conditioned` directly
   ([`src/creativity.py:650`](src/creativity.py#L650)). Neither path calls the
   scheduler-converting `pipe.forward`.

This separation is deliberate.

## Equation 16: feature map

For each level, the denoising error is:

```math
e_i(x_0) = x_0 - f(x_{t_i},t_i).
```

The finite equation-16 block is:

```math
\phi_i(x_0)
= \sqrt{\frac{\Delta\gamma_i}{G}}\,e_i(x_0).
```

The final feature map concatenates all flattened blocks:

```math
\Phi(x_0)
= \operatorname{concat}_{i=1}^{G}
  \left[
    \sqrt{\frac{\Delta\gamma_i}{G}}\,e_i(x_0)
  \right].
```

Error, weight, flattening, and concatenation are implemented at
[`src/creativity.py:308`](src/creativity.py#L308).

For 512x512 FLUX.1-schnell output, the expected packed endpoint shape is:

```text
(batch, 1024, 64)
```

With `G=64`, one feature vector has:

```text
64 * 1024 * 64 = 4,194,304 values
```

This is why candidate IEM differentiation is memory- and compute-intensive.

### Dtypes

- FLUX is loaded in bfloat16
  ([`src/flux_pipeline.py:77`](src/flux_pipeline.py#L77)).
- Endpoint/noisy latent input is cast to the endpoint dtype before the
  transformer call.
- Noise construction, equation 14 residuals, and equation 16 weights use
  float32 ([`src/creativity.py:295`](src/creativity.py#L295)).
- Reference sums and squared norms use float64 for numerical stability
  ([`src/creativity.py:333`](src/creativity.py#L333)).
- `mu_omega` is stored in float32; scalar `v_omega` remains float64
  ([`src/creativity.py:378`](src/creativity.py#L378)).
- The reward returned to Psi is float32
  ([`src/creativity.py:754`](src/creativity.py#L754)).

## Equation 21: reference statistics and reward

For one noise table, let the reference features be
`Phi(x'_1), ..., Phi(x'_M)`. The default is `M=48`; the current smoke run uses
`M=3`.

The implementation accumulates only:

```math
M,
\qquad
\sum_{j=1}^{M}\Phi(x'_j),
\qquad
\sum_{j=1}^{M}\|\Phi(x'_j)\|^2.
```

See [`src/creativity.py:319`](src/creativity.py#L319). It then computes:

```math
\mu_\omega
= \frac{1}{M}\sum_{j=1}^{M}\Phi(x'_j),
```

and:

```math
v_\omega
= \frac{1}{M}\sum_{j=1}^{M}\|\Phi(x'_j)\|^2
  - \|\mu_\omega\|^2.
```

See [`src/creativity.py:361`](src/creativity.py#L361).

`v_omega` is not the mean of `||Phi(x'_j)-mu_omega||^2` in the code. It is
implemented directly in the equation-21 form above. Those expressions are
mathematically equal, but storing the raw first moment and mean squared norm
keeps the implementation aligned with the paper.

`clamp_min(0)` at [`src/creativity.py:381`](src/creativity.py#L381) only guards
against a tiny negative scalar caused by floating-point cancellation. In exact
arithmetic, `v_omega >= 0`.

For candidate feature `Phi(x)`, the reward is:

```math
R_{\mathrm{IEM}}(x)
= \|\Phi(x)-\mu_\omega\|^2 + v_\omega.
```

See [`src/creativity.py:385`](src/creativity.py#L385).

Equivalently:

```math
R_{\mathrm{IEM}}(x)
= \frac{1}{M}\sum_{j=1}^{M}
  \|\Phi(x)-\Phi(x'_j)\|^2.
```

Thus each candidate is compared to **all 48 references**, but not by
recomputing 48 pairwise differences on every reward call. The `mu_omega` and
`v_omega` selected for the active noise table summarize that table's full
reference cloud exactly for this squared-distance reward.

There is no additional score normalization. Psi's existing `alpha_mcmc`,
`alpha`, `grad_norm`, and `smc_grad_norm` control reward/gradient influence.

## Candidate gradient path

The reward input is a latent Tweedie endpoint, not a decoded RGB image.
`reward_input_type: latent` is configured at
[`config/creativity.yaml:74`](config/creativity.yaml#L74).

At the start of each reward round, the pipeline invokes
`begin_reward_evaluation` before minibatch splitting. It selects the active
table once for the whole round
([`src/flux_pipeline.py:548`](src/flux_pipeline.py#L548),
[`src/creativity.py:525`](src/creativity.py#L525)). Then, for each reward
minibatch:

1. Detach the current sampler particle and re-enable gradients on that input
   ([`src/flux_pipeline.py:559`](src/flux_pipeline.py#L559)).
2. Run the outer denoiser and generalized Tweedie calculation
   ([`src/flux_pipeline.py:563`](src/flux_pipeline.py#L563)).
3. Pass the latent Tweedie endpoint directly to `IEMReward`
   ([`src/flux_pipeline.py:572`](src/flux_pipeline.py#L572)).
4. Build candidate equation-16 features using `pipe.predict`
   ([`src/creativity.py:744`](src/creativity.py#L744)).
5. Compute equation 21.
6. Differentiate the scalar reward back to the current sampler particle with
   `torch.autograd.grad`
   ([`src/flux_pipeline.py:602`](src/flux_pipeline.py#L602)).
7. Apply the configured gradient norm cap
   ([`src/flux_pipeline.py:607`](src/flux_pipeline.py#L607)).

The complete path is:

```text
Psi particle
  -> outer FLUX velocity
  -> Tweedie x_0
  -> noisy IEM probes x_t
  -> native FLUX probe velocities
  -> equation-14 clean predictions
  -> equation-16 Phi(x_0)
  -> equation-21 reward
  -> gradient with respect to the Psi particle
```

Reference feature tensors, `mu_omega`, and `v_omega` do not require gradients.
The transformer parameters are frozen, but autograd still differentiates
through transformer operations with respect to their latent inputs. This is
verified in
[`tests/test_creativity.py:246`](tests/test_creativity.py#L246).

Candidate probe calls use non-reentrant activation checkpointing by default.
Instead of retaining the frozen FLUX transformer's activations for all 64
levels until equation 21 is differentiated, backward recomputes each probe
chunk. This preserves the feature values and candidate gradients while bounding
peak activation memory. It increases denoiser compute during backward and does
not apply to no-grad reference features. The full H200 launcher additionally
uses `level_batch_size=1`; this changes only probe batching, not the schedule,
noise, feature definition, or reward.

### Decode behavior

IEM does not use decoded images for its score. However, the current shared
pipeline still decodes each Tweedie after latent reward evaluation, under
`torch.no_grad()`, so it can return visualization tensors
([`src/flux_pipeline.py:575`](src/flux_pipeline.py#L575)).

This decode is outside the reward graph, but it can still be a substantial
runtime cost. The timing field `reward_decode_seconds` measures it. Removing
unrequested decodes would be a valid future optimization, provided
`save_tweedies` and existing visualization behavior remain correct.

## Is Psi iterative, and how often is IEM evaluated?

Yes. Each generated image passes through two iterative reward-guided stages.

### pCNL MCMC

pCNL performs:

1. one reward/gradient evaluation at the initial position;
2. one evaluation for every proposal in
   `num_mcmc_steps + burn_in` iterations;
3. one final evaluation after the loop.

The calls are visible at
[`src/mcmc.py:289`](src/mcmc.py#L289),
[`src/mcmc.py:292`](src/mcmc.py#L292), and
[`src/mcmc.py:321`](src/mcmc.py#L321).

Therefore:

```text
MCMC reward rounds = num_mcmc_steps + burn_in + 2
```

With full defaults:

```text
25 + 25 + 2 = 52 reward rounds
```

Each round contains five chains. Because `grad_minibatch_size=1`, the pipeline
splits those chains into five separate reward-model calls:

```text
52 rounds * 5 chains = 260 IEM reward-model calls
```

### SMC

SMC performs one initial reward/gradient evaluation
([`src/runner.py:105`](src/runner.py#L105)), then one reward evaluation after
each inference step
([`src/runner.py:127`](src/runner.py#L127),
[`src/runner.py:171`](src/runner.py#L171)). The final evaluation asks only for
the value, not its gradient
([`src/runner.py:183`](src/runner.py#L183)).

Therefore:

```text
SMC reward rounds = num_inference_steps + 1
```

With full defaults:

```text
25 + 1 = 26 reward rounds
```

Each round contains ten particles and is split into ten calls:

```text
26 rounds * 10 particles = 260 IEM reward-model calls
```

### Full/default total per generated image

| Quantity | MCMC | SMC | Total |
| --- | ---: | ---: | ---: |
| Reward rounds | 52 | 26 | **78** |
| IEM reward-model calls after minibatch splitting | 260 | 260 | **520** |
| IEM denoiser probes at 64 levels | 16,640 | 16,640 | **33,280** |
| Outer Tweedie denoiser calls | 260 | 260 | **520** |

These counts assume the current defaults:

- `num_chains=5`;
- `num_particles=10`;
- `grad_minibatch_size=1`;
- `mini_batch_size=1`;
- `num_steps=64`;
- `level_batch_size=4`.

The 48 references do **not** multiply the 520 candidate calls. In fixed mode,
they were already reduced to one `mu_omega, v_omega` pair. In staged mode,
they are reduced once per table.

### Staged table schedule

The total number of reward rounds is derived from sampler configuration:

```text
total = num_mcmc_steps + burn_in + num_inference_steps + 3
```

The extra three are pCNL's initial and final evaluations plus SMC's initial
evaluation. For zero-based round `r`, table selection is:

```text
table_index = floor(r * noise_table_count / total)
```

This evenly spaces changes along the complete pCNL-plus-SMC optimization. With
the full defaults, table 0 is used for rounds 1-10 and table 7 for rounds
70-78. Eight tables means eight table windows and seven transitions after the
initial table.

The schedule resets for each candidate prompt. Seeds and assignment are
deterministic
([`src/creativity.py:517`](src/creativity.py#L517)).

### One-time full reference cost

With fixed-mode defaults and batch size one:

| Reference operation | Count |
| --- | ---: |
| Endpoint denoising | `48 * 4 = 192` FLUX steps |
| IEM feature probes | `48 * 64 = 3,072` FLUX probes |
| Total reference transformer evaluations | **3,264** |

This cost occurs once per invocation and is amortized across all candidate
images in the dataset.

With `noise_table_mode=staged noise_table_count=8`, endpoint generation stays
at 192 steps because the same 48 endpoints are reused. Reference IEM probes
become:

```text
48 references * 64 levels * 8 tables = 24,576 probes
```

That eightfold reference-feature cost is paid once before candidate
optimization. Candidate cost stays at 64 probes per minibatch; only the active
table changes.

## Current H200 smoke run

The current launcher overrides the full defaults at
[`run_creativity_smoke_h200.slurm:62`](experiment_scripts/run_creativity_smoke_h200.slurm#L62):

| Parameter | Smoke value |
| --- | ---: |
| Candidate prompts | 1 (`cat`) |
| References | 3, one per source |
| Reference endpoint steps | 4 |
| IEM levels | 2 |
| pCNL chains | 1 |
| pCNL retained steps | 1 |
| pCNL burn-in | 1 |
| SMC particles | 1 |
| SMC inference steps | 2 |
| Gradient minibatch | 1 |
| Baseline comparison | enabled |
| Timing | enabled |
| Noise-table mode | fixed, one table |

For that run:

```text
MCMC reward rounds = 1 + 1 + 2 = 4
SMC reward rounds  = 2 + 1     = 3
Total reward calls = 7
IEM probe calls    = 7 * 2     = 14
Outer denoisers    = 7
```

Its one-time reference work is:

```text
3 references * 4 endpoint steps = 12 endpoint denoiser steps
3 references * 2 IEM levels     = 6 reference IEM probes
```

The optional matched baseline adds four vanilla FLUX denoising steps. The
already-running smoke process was launched with this fixed-table configuration
and is not changed by source edits made after launch.

## Conditioning

References have different prompts within a batch. Their prompt and pooled
embeddings are repeated in the same level-major order used when IEM flattens
`(levels, endpoints, ...)`
([`src/creativity.py:214`](src/creativity.py#L214)). The batch-aligned native
velocity helper is
[`src/flux_pipeline.py:269`](src/flux_pipeline.py#L269).

Candidates use the currently encoded candidate prompt through
`pipe.predict` ([`src/flux_pipeline.py:316`](src/flux_pipeline.py#L316)).

FLUX.1-schnell has no guidance embedding in this pipeline, and the config uses
`true_cfg_scale=1.0`
([`config/creativity.yaml:23`](config/creativity.yaml#L23)). Therefore this run
does not perform classifier-free guidance.

## Baseline comparison

When baseline comparison is enabled, the raw latent is cloned before pCNL
modifies it
([`src/flux_pipeline.py:225`](src/flux_pipeline.py#L225)). The baseline then
uses:

- the same candidate prompt embeddings;
- the first preserved raw initial latent;
- vanilla FLUX.1-schnell;
- `baseline_num_inference_steps`, default 4.

See [`src/flux_pipeline.py:245`](src/flux_pipeline.py#L245). The Psi image and
baseline have prompt-bearing names such as:

```text
00000_cat_77af778b.png
00000_cat_77af778b_baseline.png
```

Saving is implemented at [`main.py:129`](main.py#L129).

## Timing

Set `log_timing=true` to synchronize CUDA around measured stages.

Reference timing reports:

- prompt sampling;
- prompt encoding;
- endpoint generation;
- reference IEM features;
- reference statistics;
- total reference preparation.

See [`src/creativity.py:710`](src/creativity.py#L710).

Candidate-level timing reports:

- model/reward setup;
- candidate prompt encoding;
- MCMC plus initialization;
- SMC;
- final decode;
- image save;
- matched baseline generation;
- total candidate time.

See [`main.py:30`](main.py#L30).

The reward pipeline separately accumulates:

- `outer_velocity`;
- `outer_tweedie`;
- `reward_forward`;
- `reward_decode`;
- `reward_backward`;
- `reward_gradient_scale`.

See [`src/flux_pipeline.py:119`](src/flux_pipeline.py#L119) and
[`src/flux_pipeline.py:562`](src/flux_pipeline.py#L562).

The IEM reward also logs each candidate feature calculation separately from
the inexpensive equation-21 reduction
([`src/creativity.py:758`](src/creativity.py#L758)).

## Configuration reference

### IEM and reference parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `sigma_min` | 1 | Smallest VE sigma boundary. |
| `sigma_max` | 1000 | Largest VE sigma boundary. |
| `num_steps` | 64 | Number of equation-16 integration intervals. |
| `level_batch_size` | 4 | Levels assembled per `iem_features` chunk; matches the RAM default. |
| `checkpoint_candidate_features` | `true` | Recompute candidate probe denoisers during backward instead of retaining every level's transformer activations. |
| `noise_table_mode` | `fixed` | `fixed` uses one table; `staged` changes tables across reward-round windows. |
| `noise_table_count` | 1 | Tables prepared and used. Set to 8 with `staged` for eight windows. |
| `reference_sample_count` | 48 | Total fixed reference endpoints. |
| `reference_batch_size` | 1 | Prompts/endpoints processed together during reference preparation. |
| `reference_num_inference_steps` | 4 | Vanilla FLUX steps per reference endpoint. |
| `reference_seed` | 30,000,000 | Base seed for prompt selection and endpoint generators. |
| `noise_seed` | 90,000,000 | Base seed; table `k` adds `k`. |
| `reward_input_type` | `latent` | Sends Tweedie latents, not decoded images, to IEM. |

The authoritative values are
[`config/creativity.yaml:70`](config/creativity.yaml#L70).

### Reward-guided sampler parameters

| Parameter | Default | Role |
| --- | ---: | --- |
| `alpha_mcmc` | 0.1 | Reward/KL scale in pCNL acceptance and proposals. |
| `num_mcmc_steps` | 25 | Number of post-burn-in MCMC iterations retained. |
| `burn_in` | 25 | Initial MCMC iterations discarded. |
| `num_chains` | 5 | Parallel pCNL chains. |
| `grad_norm` | 0.0004 | MCMC reward-gradient RMS cap. |
| `alpha` | 0.1 | Reward/KL scale in SMC. |
| `num_particles` | 10 | SMC particles selected from pCNL. |
| `num_inference_steps` | 25 | SMC denoising steps. |
| `smc_grad_norm` | 0.0004 | SMC reward-gradient RMS cap. |
| `ess_threshold` | 0.5 | Resampling threshold as a particle-count fraction. |

See [`config/creativity.yaml:36`](config/creativity.yaml#L36) and
[`config/creativity.yaml:54`](config/creativity.yaml#L54).

## Existing reward behavior

The IEM integration adds an explicit latent-reward branch. Rewards without
`reward_input_type` default to `"image"` and still receive decoded images
([`src/flux_pipeline.py:553`](src/flux_pipeline.py#L553)).

Regression tests verify:

- image rewards still receive decoded tensors
  ([`tests/test_creativity.py:356`](tests/test_creativity.py#L356));
- latent rewards receive Tweedie latents before detached decode
  ([`tests/test_creativity.py:387`](tests/test_creativity.py#L387));
- reference computation has no gradients while candidate computation does
  ([`tests/test_creativity.py:246`](tests/test_creativity.py#L246));
- equation 21 equals direct mean pairwise squared distance
  ([`tests/test_creativity.py:65`](tests/test_creativity.py#L65));
- shared noise is deterministic
  ([`tests/test_creativity.py:116`](tests/test_creativity.py#L116));
- staged tables cover all 78 default rounds and advance only once before
  minibatch splitting
  ([`tests/test_creativity.py:136`](tests/test_creativity.py#L136),
  [`tests/test_creativity.py:419`](tests/test_creativity.py#L419));
- balanced exclusions are enforced
  ([`tests/test_creativity.py:170`](tests/test_creativity.py#L170)).

## Relationship to the RAM implementation

The schedule, equation-14 clean prediction, equation-16 feature construction,
and corrected equation-21 first-moment/squared-norm calculation follow the
validated implementation in the sibling `unsupervised_creativity_ram`
repository.

The Psi adaptation deliberately does **not** port:

- the dynamic reference reservoir;
- epoch refreshes;
- priority sampling;
- distributed reference-statistic merging;
- training checkpoint state for IEM references.

RAM's default eight tables are assigned across candidate prompt groups within
an epoch. Psi's optional eight-table mode instead assigns tables across
successive reward rounds for one candidate. Both designs enforce the same
essential invariant: candidate and reference features in an equation-21 score
use the same table. Psi keeps the reference endpoints fixed rather than using
RAM's reservoir and epoch refresh.

This repository uses frozen **FLUX.1-schnell**, not SD3.5 Medium and not the
RAM CFG-distillation baseline LoRA. The IEM equations remain applicable because
FLUX uses the same linear rectified-flow velocity parameterization required by
`f=x_t-t*v`.

## Important invariants and limitations

- The implementation is currently single-process/single-GPU. `main.py` uses
  `cuda:0` ([`main.py:27`](main.py#L27)).
- References are fixed only for one process invocation; there is no checkpoint
  for reference statistics.
- All candidates in an invocation must have the same packed latent shape as
  the reference noise tables.
- Higher IEM is treated as higher reward: it means a larger mean squared
  feature distance from the reference cloud.
- The reference model and candidate IEM probe model are the same frozen
  FLUX.1-schnell transformer used by Psi.
- The feature map is large, and differentiating through all probe denoisers is
  the expected dominant cost at full settings.
- `level_batch_size` can change batching and memory use, but not the selected
  sigmas, noise tables, feature definition, or equation-21 result.
- Disabling `checkpoint_candidate_features` at 64 levels can retain enough
  transformer activations to exhaust even a 140 GB H200.
