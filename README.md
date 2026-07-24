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
- `--save_baseline_comparison` : Save a vanilla FLUX image as `<index>_baseline.png` from the same raw noise used to initialize Ψ-Sampler. Disabled by default.
- `--baseline_num_inference_steps 4` : Number of vanilla FLUX denoising steps used for the optional baseline image.

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
