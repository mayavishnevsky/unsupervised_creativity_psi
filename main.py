import os
import argparse
from dataclasses import dataclass
from PIL import ImageDraw
from tqdm import tqdm
import json

from src.utils import *
from src.flux_pipeline import StochasticFluxPipeline, TimeSampler
from src.runner import *
from src.reward_model import *
import src.creativity  # Registers the optional IEM reward.
from src.mcmc import pCNL


@ignore_kwargs
@dataclass
class Config:
    seed: int = 0
    negative_prompt: str = None
    height: int = 512
    width: int = 512
    save_baseline_comparison: bool = False
    baseline_num_inference_steps: int = 4
    log_timing: bool = False

def main(main_cfg, CFG, args, task_name):
    device = torch.device("cuda:0")

    def timing_start():
        if not main_cfg.log_timing:
            return None
        return synchronized_time(device)

    def log_timing(stage, start, **values):
        if start is None:
            return
        elapsed = synchronized_time(device) - start
        details = " ".join(f"{key}={value}" for key, value in values.items())
        print(
            f"[timing] stage={stage} seconds={elapsed:.3f} {details}".rstrip(),
            flush=True,
        )

    setup_start = timing_start()
    with suppress_print():
        time_sampler = TimeSampler(device, CFG)
        pipe = StochasticFluxPipeline(device, CFG)
        reward_model = get_reward_model(task_name)(torch.float32, device, args.save_dir, CFG)
        SMC_runner = SMC(CFG)
    log_timing("model_and_reward_setup", setup_start)

    if args.data_path.endswith(".json"):
        dataset = json.load(open(args.data_path, 'r'))
    else:
        with open(args.data_path, 'r') as f:
            dataset = [line.strip() for line in f if line.strip() != '']

    if hasattr(reward_model, "prepare_references"):
        dataset_values = dataset if isinstance(dataset, list) else dataset.values()
        candidate_prompts = [
            data if isinstance(data, str) else data["prompt"]
            for data in dataset_values
        ]
        pipe.load_encoder()
        reward_model.prepare_references(
            pipe,
            excluded_prompts=candidate_prompts,
        )
        pipe.unload_encoder()
    
    for idx, data in enumerate(tqdm(dataset, total=len(dataset), desc="Benchmark")):
        candidate_start = timing_start()
        data = data if isinstance(dataset, list) else dataset[data]
        prompt = data if isinstance(data, str) else data["prompt"]
        phrases = data['phrases'] if isinstance(data, dict) and "phrases" in data else None
        output_stem = prompt_output_stem(idx, prompt)

        stage_start = timing_start()
        pipe.load_encoder()
        pipe.encode_prompt(prompt, main_cfg.negative_prompt, phrases=phrases)
        reward_model.register_data(data)
        pipe.unload_encoder()
        log_timing(
            "candidate_prompt_encoding",
            stage_start,
            candidate=idx,
            output_stem=output_stem,
        )

        sample_seed = main_cfg.seed + idx
        seed_everything(sample_seed)
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        
        reward_model.cfg.grad_const_scale = CFG.grad_const_scale
        reward_model.cfg.grad_norm = CFG.grad_norm
        
        # MCMC
        pipe.reset_timing()
        stage_start = timing_start()
        prepared_latents = pipe.prepare_latents(
            height=main_cfg.height,
            width=main_cfg.width,
            reward_model=reward_model,
            generator=generator,
            return_initial_latents=main_cfg.save_baseline_comparison,
        )
        if main_cfg.save_baseline_comparison:
            latents, initial_latents = prepared_latents
        else:
            latents = prepared_latents
        log_timing("candidate_mcmc_and_initialization", stage_start, candidate=idx)
        pipe.print_timing(f"candidate={idx} phase=mcmc")

        reward_model.cfg.grad_norm = CFG.smc_grad_norm
        reward_model.cfg.grad_const_scale = CFG.smc_grad_const_scale

        # SMC
        pipe.reset_timing()
        stage_start = timing_start()
        sample, sample_reward = SMC_runner.run(pipe, time_sampler, reward_model, latents, generator, idx=idx)
        log_timing("candidate_smc", stage_start, candidate=idx)
        pipe.print_timing(f"candidate={idx} phase=smc")

        stage_start = timing_start()
        final_latent = pipe.decode_latents(sample.detach(), output_type="pt")
        log_timing("candidate_final_decode", stage_start, candidate=idx)
        
        image = torchvision.transforms.ToPILImage()(final_latent[0].float().cpu().clamp(0, 1))
        stage_start = timing_start()
        image.save(os.path.join(args.save_dir, f"{output_stem}.png"))
        log_timing("candidate_image_save", stage_start, candidate=idx)

        if main_cfg.save_baseline_comparison:
            stage_start = timing_start()
            baseline_image = pipe.generate_baseline(
                initial_latents,
                num_inference_steps=main_cfg.baseline_num_inference_steps,
            )
            baseline_image.save(
                os.path.join(args.save_dir, f"{output_stem}_baseline.png")
            )
            log_timing("candidate_baseline_generation", stage_start, candidate=idx)

        if args.save_reward:
            draw = ImageDraw.Draw(image)
            text = f"{sample_reward.item():.5f}" if hasattr(sample_reward, "item") else f"{sample_reward:.5f}"
            draw.rectangle([0, 0, 120, 20], fill=(0, 0, 0, 128))  
            draw.text((5, 2), text, fill=(255, 255, 255))
            
            if "layout_to_image" in task_name:
                draw_box(image, data['bboxes'], phrases, main_cfg.height, main_cfg.width)
                
            image.save(
                os.path.join(args.save_dir, "img_rewards", f"{output_stem}.png")
            )
        if main_cfg.log_timing:
            log_timing(
                "candidate_total",
                candidate_start,
                candidate=idx,
                reward=float(sample_reward.detach().float().cpu()),
                output_stem=output_stem,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/layout_to_image.yaml")
    parser.add_argument("--data_path", default="./data/layout_to_image.json")
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--save_tweedies", action="store_true", help="Save the tweedies")
    parser.add_argument("--save_reward", action="store_true", help="Save the reward value on the image")
    parser.add_argument(
        "--save_baseline_comparison",
        action="store_true",
        help="Save vanilla FLUX output from the same raw initial noise",
    )
    parser.add_argument(
        "--baseline_num_inference_steps",
        type=int,
        default=None,
        help="Denoising steps for the optional vanilla FLUX baseline",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--extra_tag", default=None)


    args, extras = parser.parse_known_args()
    CFG = load_config(args.config, cli_args=extras)
    if args.save_baseline_comparison:
        CFG.save_baseline_comparison = True
    if args.baseline_num_inference_steps is not None:
        CFG.baseline_num_inference_steps = args.baseline_num_inference_steps

    task_name = args.config.split("/")[-1].split(".")[0]
    main_cfg = Config(**CFG)

    step_size = CFG.step_size
    rho = pCNL.get_rho(step_size)
    CFG.rho = rho
    grad_norm = CFG.grad_norm
    smc_grad_norm = CFG.smc_grad_norm
    mcmc_name = CFG.mcmc
        
    name = f"{mcmc_name}_test"
    
    if args.extra_tag is not None:
        name = name + "_" + args.extra_tag

    qualitative_dir = os.path.join(args.save_dir, task_name) if args.tag is None else os.path.join(args.save_dir, task_name, args.tag)
    args.save_dir = os.path.join(qualitative_dir, name)
    args.misc_dir = os.path.join(args.save_dir, "misc")
    os.makedirs(args.misc_dir, exist_ok=True)

    with open(os.path.join(args.misc_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(CFG))

    CFG.save_dir = args.save_dir
    CFG.qualitative_dir = qualitative_dir
    CFG.misc_dir = args.misc_dir
    CFG.name = name
    CFG.save_tweedies = args.save_tweedies
    CFG.reward_name = task_name
    if args.save_reward:
        os.makedirs(os.path.join(args.save_dir, "img_rewards"), exist_ok=True)
    main(main_cfg, CFG, args, task_name)
