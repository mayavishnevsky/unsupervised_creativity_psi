import unittest
from types import SimpleNamespace

import torch

from src.flux_pipeline import StochasticFluxPipeline
from src.utils import prompt_output_stem


class _FakeFluxPipeline:
    vae_scale_factor = 8
    dtype = torch.bfloat16
    transformer = SimpleNamespace(config=SimpleNamespace(in_channels=16))

    def __init__(self):
        self.baseline_kwargs = None

    def prepare_latents(self, batch_size, **kwargs):
        latents = torch.arange(batch_size * 12, dtype=torch.bfloat16).reshape(batch_size, 3, 4)
        return latents, torch.zeros(1)

    def __call__(self, **kwargs):
        self.baseline_kwargs = kwargs
        return SimpleNamespace(images=["baseline-image"])


class _MutatingInitialSampler:
    cfg = SimpleNamespace(num_chains=2)

    def __call__(self, latents, **kwargs):
        latents.add_(10)
        return latents


class BaselineComparisonTests(unittest.TestCase):
    def make_pipeline(self):
        pipeline = StochasticFluxPipeline.__new__(StochasticFluxPipeline)
        pipeline.pipe = _FakeFluxPipeline()
        pipeline.device = torch.device("cpu")
        pipeline.cfg = SimpleNamespace(
            num_particles=3,
            true_cfg_scale=1.0,
            guidance_scale=3.5,
        )
        pipeline.init_sampling_method = _MutatingInitialSampler()
        pipeline.prompt_embeds = torch.ones(1, 2, 3)
        pipeline.pooled_peompt_embeds = torch.ones(1, 3)
        pipeline.negative_prompt_embeds = None
        pipeline.negative_pooled_prompt_embeds = None
        pipeline.do_true_cfg = False
        return pipeline

    def test_prepare_latents_preserves_pre_mcmc_noise(self):
        pipeline = self.make_pipeline()

        sampled, initial = pipeline.prepare_latents(
            height=512,
            width=512,
            reward_model=object(),
            return_initial_latents=True,
        )

        torch.testing.assert_close(sampled, initial + 10)
        self.assertNotEqual(sampled.data_ptr(), initial.data_ptr())

    def test_prepare_latents_keeps_original_return_type_when_disabled(self):
        pipeline = self.make_pipeline()

        sampled = pipeline.prepare_latents(
            height=512,
            width=512,
            reward_model=object(),
        )

        self.assertIsInstance(sampled, torch.Tensor)

    def test_generate_baseline_uses_first_raw_latent(self):
        pipeline = self.make_pipeline()
        pipeline.height = 512
        pipeline.width = 512
        initial = torch.randn(2, 3, 4)

        image = pipeline.generate_baseline(initial, num_inference_steps=4)

        self.assertEqual(image, "baseline-image")
        kwargs = pipeline.pipe.baseline_kwargs
        torch.testing.assert_close(kwargs["latents"], initial[:1])
        self.assertNotEqual(kwargs["latents"].data_ptr(), initial.data_ptr())
        self.assertEqual(kwargs["num_inference_steps"], 4)
        self.assertEqual(kwargs["height"], 512)
        self.assertEqual(kwargs["width"], 512)

    def test_output_stem_contains_a_safe_prompt_and_hash(self):
        stem = prompt_output_stem(7, "A cat / wearing a red hat!")

        self.assertTrue(stem.startswith("00007_A_cat_wearing_a_red_hat_"))
        self.assertNotIn("/", stem)
        self.assertEqual(stem, prompt_output_stem(7, "A cat / wearing a red hat!"))
        self.assertLessEqual(len(stem), 5 + 1 + 96 + 1 + 8)


if __name__ == "__main__":
    unittest.main()
