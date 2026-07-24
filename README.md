<h1 align="center">Ψ-Sampler</h1>
<div align="center">
  
## Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score-Based Generative Models
## **NeurIPS 2025, Spotlight**
</div>

![teaser](assets/teaser.png)

<p align="center">
  <a href="https://arxiv.org/abs/2506.01320">
    <img src="https://img.shields.io/badge/arXiv-2506.01320-red" alt="arXiv 2506.01320" />
  </a>
  <a href="https://psi-sampler.github.io/">
    <img src="https://img.shields.io/badge/Website-psi_sampler.github.io-blue" alt="Website" />
  </a>
</p>
<!-- Authors -->
<p align="center">
  <a href="https://github.com/taehoon-yoon">Taehoon Yoon*</a>,
  <a href="https://cactus-save-5ac.notion.site/4020147bcaef4257888b08b0a4ef238d">Yunhong Min*</a>,
  <a href="https://32v.github.io/">Kyeongmin Yeo*</a>,
  <a href="https://mhsung.github.io">Minhyuk Sung</a>
  (* equal contribution)
</p>

## Introduction

We propose **Ψ-Sampler**, an SMC-based framework that improves inference-time reward alignment in score-based generative models via efficient posterior initialization using the pCNL algorithm.

[//]: # "### Abstract"

> We introduce Ψ-Sampler, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank–Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments.

<!-- Release Note -->

### Release

- **[02/02/25]** 🔢 We have released the implementation for quantity-aware generation.
- **[03/12/25]** 🔥 We have released the implementation of _Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score-Based Generative Models_ for layout-to-image generation and aesthetic-preference generation.

### Setup

Create a Conda environment:

```
conda create -n psi_sampler python=3.10 -y
conda activate psi_sampler
```

Clone this repository:

```
git clone https://github.com/KAIST-Visual-AI-Group/Psi-Sampler.git
cd Psi-Sampler
```

Install PyTorch and requirements:

```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Configuration:

- `--mcmc` : MCMC method (`mala`, `pcnl`)
- `--num_mcmc_steps` : Number of MCMC steps used for the initial particle sampling process
- `--num_chains` : Number of MCMC processes
- `--burn_in` : Number of initial samples to discard in the MCMC process
- `--alpha_mcmc` : Strength of KL regularization for the initial particle sampling process
- `--alpha` : Strength of KL regularizaiton for the subsequent SMC process
- `--num_particles` : Number of selected initial particles
- `--num_inference_steps` : Number of denoising steps in the generative process
- `--ess_threshold` : Minimum acceptable ratio for the Effective Sample Size (ESS)

### Optional Flags:

- `--save_reward` : Display the reward value on the saved images.
- `--save_tweedies` : Save the step-wise particle tweedies for each MCMC and SMC process.
- `--save_baseline_comparison` : Save a vanilla FLUX image as `<prompt-stem>_baseline.png` from the same raw noise used to initialize Ψ-Sampler. Disabled by default.
- `--baseline_num_inference_steps 4` : Number of vanilla FLUX denoising steps used for the optional baseline image.
- `log_timing=true` : Emit synchronized timings for reference preparation,
  MCMC/SMC, IEM reward evaluation, backward passes, decoding, and baseline
  generation.

Generated image names include a readable prompt slug and short prompt hash,
for example `00000_cat_77af778b.png` and
`00000_cat_77af778b_baseline.png`.

</details>

### Layout-to-Image Generation

We provide example data file for layout-to-image generation in `data/layout_to_image.json`. You can run layout-to-image generation using the following command.

You may optionally override configuration values by specifying arguments directly in the command line:

```
CUDA_VISIBLE_DEVICES={$DEVICE} python main.py --tag layout_to_image --config ./config/layout_to_image.yaml --data_path ./data/layout_to_image.json --save_dir ./results_layout_to_image --alpha_mcmc={$VALUE} --save_reward --save_tweedies
```

### Aesthetic-Preference Generation

We provide example data file for aesthetic-preference generation in `data/aesthetic.txt`. You can run aesthetic-preference generation using the following command.

You may optionally override configuration values by specifying arguments directly in the command line:

```
CUDA_VISIBLE_DEVICES={$DEVICE} python main.py --tag aesthetic --config ./config/aesthetic.yaml --data_path ./data/aesthetic.txt --save_dir ./results_aesthetic --alpha_mcmc={$VALUE} --save_reward --save_tweedies
```

### IEM Creativity Reward

`src/creativity.py` implements the latent Information Estimation Metric (IEM)
from equations 14, 16, and 21 of *Unsupervised Creative Generation*. The
candidate input is the differentiable Tweedie `x_0` estimate: gradients pass
through the IEM denoiser probes and back into the Psi particle. Reference
endpoints and their feature statistics are generated once before sampling
under `torch.no_grad()`.

The default `config/creativity.yaml`:

- draws 48 balanced reference prompts from the PickScore, OCR, and GenEval
  files in the sibling `unsupervised_creativity_ram` checkout;
- excludes every prompt in the current candidate dataset from that draw;
- generates one vanilla FLUX.1-schnell endpoint per reference with four steps;
- uses one deterministic noise table for every reference and candidate; and
- evaluates 64 log-uniform sigma intervals from 1000 down to 1.

Prompt files, reference count, seeds, sigma bounds, integration levels, and
batch sizes can be changed in YAML or with OmegaConf command-line overrides:

```
CUDA_VISIBLE_DEVICES={$DEVICE} python main.py \
  --tag creativity \
  --config ./config/creativity.yaml \
  --data_path ./data/aesthetic.txt \
  --save_dir ./results_creativity \
  reference_sample_count=48 num_steps=64
```

IEM is substantially more expensive than image-space rewards because every
reward evaluation differentiates through `num_steps` additional frozen FLUX
denoiser calls. `experiment_scripts/run_creativity_smoke_h100.slurm` provides a
small two-level, three-reference integration check.
`experiment_scripts/run_creativity_smoke_h200.slurm` runs that check on H200
with timing and matched baseline output enabled.

### Quantity-Aware Generation

This task requires checkpoints from [T2ICount](https://github.com/cha15yq/T2ICount). Download the following files to `misc/t2icount/`:

- [v1-5-pruned-emaonly.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt)
- [T2ICount checkpoint](https://drive.google.com/file/d/1lw5LgpYP7vTazaMWTgNa6nFoZ63j-st9/view)

You may optionally override configuration values by specifying arguments directly in the command line:

```
CUDA_VISIBLE_DEVICES={$DEVICE} python main.py --tag compile --config ./config/quantity_aware.yaml --data_path ./data/quantity_aware.json --save_dir ./results_quantity_aware --alpha_mcmc={$VALUE} --save_reward --save_tweedies
```

### Acknowledgement

We borrow codes from [T2ICount](https://github.com/cha15yq/T2ICount) for quantity-aware generation. Many thanks to the authors for sharing their codes.

## Citation

```
@article{yoon2025psi,
  title={Psi-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models},
  author={Yoon, Taehoon and Min, Yunhong and Yeo, Kyeongmin and Sung, Minhyuk},
  journal={arXiv preprint arXiv:2506.01320},
  year={2025}
}
```
