import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.creativity import (
    BalancedPromptSampler,
    IEMReward,
    expected_psi_reward_rounds,
    iem_features,
    iem_reward_from_reference_statistics,
    iem_schedule_terms,
    iem_sigma_schedule,
    reference_statistics,
    repeat_endpoint_conditions,
    sample_iem_noise_table,
    staged_noise_table_index,
    update_reference_sums,
)
from src.flux_pipeline import StochasticFluxPipeline


class FormulaTests(unittest.TestCase):
    def test_sigma_schedule_and_gamma_widths(self):
        schedule = iem_sigma_schedule(1.0, 100.0, 2)
        torch.testing.assert_close(schedule, torch.tensor([100.0, 10.0, 1.0]))

        probes, delta_gamma = iem_schedule_terms(schedule)
        torch.testing.assert_close(probes, schedule[:-1])
        torch.testing.assert_close(
            delta_gamma,
            torch.tensor([0.01 - 0.0001, 1.0 - 0.01]),
        )

    def test_equations_14_and_16(self):
        x_0 = torch.tensor([[1.0, -2.0]])
        schedule = torch.tensor([2.0, 1.0])
        noise = torch.tensor([[[3.0, 1.0]]])

        features = iem_features(
            x_0,
            lambda x_t, t: torch.zeros_like(x_t),
            schedule,
            noise,
        )

        flow_time = 2.0 / 3.0
        x_t = (x_0 + 2.0 * noise[0]) / 3.0
        expected = (0.75**0.5) * (x_0 - x_t)
        torch.testing.assert_close(features, expected)

        def perfect_velocity(x_t, t):
            repeated_x_0 = x_0.repeat(x_t.shape[0] // x_0.shape[0], 1)
            return (x_t - repeated_x_0) / t[:, None]

        perfect_features = iem_features(
            x_0,
            perfect_velocity,
            schedule,
            noise,
        )
        torch.testing.assert_close(perfect_features, torch.zeros_like(features))
        self.assertEqual(flow_time, 2.0 / 3.0)

    def test_equation_21_matches_direct_pairwise_mean(self):
        references = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        count, sum_phi, sum_norm = update_reference_sums(
            0, None, None, references[:1]
        )
        count, sum_phi, sum_norm = update_reference_sums(
            count, sum_phi, sum_norm, references[1:]
        )
        mu_omega, v_omega = reference_statistics(count, sum_phi, sum_norm)

        candidates = torch.tensor([[2.0, 3.0], [0.0, 0.0]])
        scores = iem_reward_from_reference_statistics(
            candidates, mu_omega, v_omega
        )
        direct = (
            candidates[:, None, :] - references[None, :, :]
        ).square().sum(dim=2).mean(dim=1)

        torch.testing.assert_close(scores, direct.to(torch.float64))
        torch.testing.assert_close(mu_omega, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(v_omega, torch.tensor(2.0, dtype=torch.float64))

    def test_candidate_features_are_differentiable(self):
        x_0 = torch.tensor([[0.5, -1.0]], requires_grad=True)
        schedule = iem_sigma_schedule(1.0, 4.0, 2)
        noise = sample_iem_noise_table(
            schedule,
            x_0.shape[1:],
            device="cpu",
            dtype=torch.float32,
            seed=7,
        )
        features = iem_features(
            x_0,
            lambda x_t, t: 0.25 * x_t,
            schedule,
            noise,
        )
        score = iem_reward_from_reference_statistics(
            features,
            torch.zeros(features.shape[1]),
            0.0,
        ).sum()
        score.backward()

        self.assertIsNotNone(x_0.grad)
        self.assertTrue(torch.isfinite(x_0.grad).all())
        self.assertGreater(x_0.grad.abs().sum().item(), 0.0)

    def test_noise_table_is_fixed_and_shared_across_endpoints(self):
        schedule = iem_sigma_schedule(1.0, 10.0, 3)
        first = sample_iem_noise_table(
            schedule,
            (2, 3),
            device="cpu",
            dtype=torch.float32,
            seed=11,
        )
        second = sample_iem_noise_table(
            schedule,
            (2, 3),
            device="cpu",
            dtype=torch.float32,
            seed=11,
        )

        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (3, 1, 2, 3))

    def test_staged_noise_tables_cover_eight_windows_over_default_rounds(self):
        total = expected_psi_reward_rounds(25, 25, 25)
        assignments = [
            staged_noise_table_index(index, total, 8) for index in range(total)
        ]

        self.assertEqual(total, 78)
        self.assertEqual(sorted(set(assignments)), list(range(8)))
        self.assertTrue(
            all(left <= right for left, right in zip(assignments, assignments[1:]))
        )
        self.assertEqual(
            [assignments.count(index) for index in range(8)],
            [10, 10, 10, 9, 10, 10, 10, 9],
        )

    def test_conditions_repeat_in_level_major_order(self):
        prompt = torch.tensor([[[1.0]], [[2.0]]])
        pooled = torch.tensor([[10.0], [20.0]])
        repeated_prompt, repeated_pooled = repeat_endpoint_conditions(
            prompt, pooled, flattened_batch_size=6
        )

        torch.testing.assert_close(
            repeated_prompt[:, 0, 0],
            torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
        )
        torch.testing.assert_close(
            repeated_pooled[:, 0],
            torch.tensor([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]),
        )


class PromptSamplerTests(unittest.TestCase):
    def test_balanced_sampling_excludes_candidate_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.txt"
            second = Path(directory) / "second.txt"
            first.write_text("candidate\na1\na2\na3\n")
            second.write_text("candidate\nb1\nb2\nb3\n")
            sampler = BalancedPromptSampler([first, second])

            records = sampler.sample(
                4,
                seed=5,
                excluded_prompts=["candidate"],
            )
            texts = [record.text for record in records]

            self.assertNotIn("candidate", texts)
            self.assertEqual(sum(text.startswith("a") for text in texts), 2)
            self.assertEqual(sum(text.startswith("b") for text in texts), 2)
            self.assertEqual(
                texts,
                [
                    record.text
                    for record in sampler.sample(
                        4, seed=5, excluded_prompts=["candidate"]
                    )
                ],
            )


class _FakeBasePipeline:
    vae_scale_factor = 8

    def __init__(self):
        self.reference_grad_modes = []
        self.transformer = torch.nn.Linear(1, 1)

    def encode_prompt(self, prompt, **kwargs):
        del kwargs
        values = torch.arange(1, len(prompt) + 1, dtype=torch.float32)
        return (
            values[:, None, None],
            values[:, None],
            torch.zeros(1, 3),
        )

    def __call__(self, prompt_embeds, **kwargs):
        del kwargs
        self.reference_grad_modes.append(torch.is_grad_enabled())
        values = prompt_embeds[:, 0, 0]
        endpoints = values[:, None, None].expand(-1, 2, 2).contiguous()
        return SimpleNamespace(images=endpoints)

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        del batch_size, height, width
        return torch.zeros(1, 3, device=device, dtype=dtype)


class _FakeWrapper:
    def __init__(self):
        self.pipe = _FakeBasePipeline()
        self.cfg = SimpleNamespace(guidance_scale=3.5)
        self.reference_predict_grad_modes = []
        self.candidate_predict_grad_modes = []

    def predict_conditioned(self, x_t, flow_time, **kwargs):
        del flow_time, kwargs
        self.reference_predict_grad_modes.append(torch.is_grad_enabled())
        return 0.1 * x_t

    def predict(self, x_t, flow_time):
        del flow_time
        self.candidate_predict_grad_modes.append(torch.is_grad_enabled())
        return 0.1 * x_t


class RewardIntegrationTests(unittest.TestCase):
    def test_references_are_frozen_and_candidates_keep_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.txt"
            second = Path(directory) / "second.txt"
            first.write_text("excluded\na1\na2\n")
            second.write_text("excluded\nb1\nb2\n")
            reward = IEMReward(
                torch.float32,
                torch.device("cpu"),
                directory,
                {
                    "reference_prompt_files": [str(first), str(second)],
                    "reference_sample_count": 2,
                    "reference_batch_size": 2,
                    "reference_num_inference_steps": 1,
                    "num_steps": 2,
                    "height": 32,
                    "width": 32,
                },
            )
            pipe = _FakeWrapper()

            reward.prepare_references(pipe, excluded_prompts=["excluded"])

            self.assertTrue(reward.prepared)
            self.assertEqual(reward.reference_count, 2)
            self.assertEqual(pipe.pipe.reference_grad_modes, [False])
            self.assertTrue(
                all(mode is False for mode in pipe.reference_predict_grad_modes)
            )
            self.assertFalse(pipe.pipe.transformer.training)
            self.assertTrue(
                all(
                    parameter.requires_grad is False
                    for parameter in pipe.pipe.transformer.parameters()
                )
            )
            self.assertFalse(reward.mu_omegas.requires_grad)
            self.assertFalse(reward.v_omegas.requires_grad)

            candidate = torch.randn(1, 2, 2, requires_grad=True)
            reward.register_data("candidate")
            reward.begin_reward_evaluation()
            score = reward(candidate, pipe)
            score.sum().backward()

            self.assertTrue(all(pipe.candidate_predict_grad_modes))
            self.assertIsNotNone(candidate.grad)
            self.assertTrue(torch.isfinite(candidate.grad).all())
            with self.assertRaisesRegex(RuntimeError, "already been prepared"):
                reward.prepare_references(pipe, excluded_prompts=[])

    def test_staged_mode_prepares_matched_statistics_for_each_noise_table(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.txt"
            prompts.write_text("a\nb\n")
            reward = IEMReward(
                torch.float32,
                torch.device("cpu"),
                directory,
                {
                    "reference_prompt_files": [str(prompts)],
                    "reference_sample_count": 2,
                    "reference_batch_size": 2,
                    "reference_num_inference_steps": 1,
                    "num_steps": 2,
                    "height": 32,
                    "width": 32,
                    "noise_table_mode": "staged",
                    "noise_table_count": 3,
                    "num_mcmc_steps": 1,
                    "burn_in": 0,
                    "num_inference_steps": 1,
                },
            )
            pipe = _FakeWrapper()

            reward.prepare_references(pipe, excluded_prompts=[])

            self.assertEqual(reward.noise_tables.shape[0], 3)
            self.assertEqual(reward.mu_omegas.shape[0], 3)
            self.assertEqual(reward.v_omegas.shape, (3,))
            self.assertEqual(len(pipe.reference_predict_grad_modes), 3)
            self.assertFalse(torch.equal(reward.noise_tables[0], reward.noise_tables[1]))

            reward.register_data("candidate")
            active_tables = []
            for _ in range(reward.total_reward_rounds):
                reward.begin_reward_evaluation()
                active_tables.append(reward.active_noise_table_index)
            self.assertEqual(active_tables, [0, 0, 1, 1, 2])
            with self.assertRaisesRegex(RuntimeError, "more reward rounds"):
                reward.begin_reward_evaluation()

    def _make_reward_gradient_pipeline(self):
        pipeline = StochasticFluxPipeline.__new__(StochasticFluxPipeline)
        pipeline.cfg = SimpleNamespace(grad_minibatch_size=1)
        pipeline.forward = lambda latents, t: torch.zeros_like(latents)
        pipeline.get_tweedie = lambda latents, velocity, t: 2.0 * latents
        pipeline.decode_grad_modes = []

        def decode(latents, output_type):
            self.assertEqual(output_type, "pt")
            pipeline.decode_grad_modes.append(torch.is_grad_enabled())
            return latents + 10.0

        pipeline.decode_latents = decode
        pipeline.decode_latents_no_normalize = lambda latents: latents + 20.0
        return pipeline

    def test_image_rewards_still_receive_decoded_tensors(self):
        pipeline = self._make_reward_gradient_pipeline()

        class ImageReward:
            cfg = SimpleNamespace(
                decode_to_unnormalized=False,
                grad_norm=None,
                grad_const_scale=None,
            )

            def __init__(self):
                self.inputs = []

            def __call__(self, images, pipe):
                del pipe
                self.inputs.append(images.detach().clone())
                return images.flatten(1).sum(dim=1)

        reward = ImageReward()
        latents = torch.ones(1, 2, 2)
        _, gradients, _, decoded = pipeline.get_reward_grad_vel_tweedies(
            latents,
            reward,
            torch.ones(1, dtype=torch.float32),
        )

        torch.testing.assert_close(reward.inputs[0], torch.full((1, 2, 2), 12.0))
        torch.testing.assert_close(decoded, torch.full((1, 2, 2), 12.0))
        torch.testing.assert_close(gradients, torch.full((1, 2, 2), 2.0))
        self.assertEqual(pipeline.decode_grad_modes, [True])

    def test_latent_rewards_receive_tweedie_before_detached_decode(self):
        pipeline = self._make_reward_gradient_pipeline()

        class LatentReward:
            cfg = SimpleNamespace(
                reward_input_type="latent",
                decode_to_unnormalized=False,
                grad_norm=None,
                grad_const_scale=None,
            )

            def __init__(self):
                self.inputs = []

            def __call__(self, latents, pipe):
                del pipe
                self.inputs.append(latents.detach().clone())
                return latents.flatten(1).square().sum(dim=1)

        reward = LatentReward()
        latents = torch.ones(1, 2, 2)
        _, gradients, _, decoded = pipeline.get_reward_grad_vel_tweedies(
            latents,
            reward,
            torch.ones(1, dtype=torch.float32),
        )

        torch.testing.assert_close(reward.inputs[0], torch.full((1, 2, 2), 2.0))
        torch.testing.assert_close(decoded, torch.full((1, 2, 2), 12.0))
        torch.testing.assert_close(gradients, torch.full((1, 2, 2), 8.0))
        self.assertEqual(pipeline.decode_grad_modes, [False])

    def test_reward_round_hook_runs_once_before_minibatch_splitting(self):
        pipeline = self._make_reward_gradient_pipeline()

        class RoundAwareReward:
            cfg = SimpleNamespace(
                reward_input_type="latent",
                decode_to_unnormalized=False,
                grad_norm=None,
                grad_const_scale=None,
            )

            def __init__(self):
                self.round_count = 0
                self.tables_seen = []

            def begin_reward_evaluation(self):
                self.round_count += 1

            def __call__(self, latents, pipe):
                del pipe
                self.tables_seen.append(self.round_count)
                return latents.flatten(1).square().sum(dim=1)

        reward = RoundAwareReward()
        pipeline.get_reward_grad_vel_tweedies(
            torch.ones(2, 2, 2),
            reward,
            torch.ones(2, dtype=torch.float32),
        )

        self.assertEqual(reward.round_count, 1)
        self.assertEqual(reward.tables_seen, [1, 1])


class ConditionedPredictionTests(unittest.TestCase):
    def test_conditioned_prediction_keeps_batch_alignment(self):
        calls = []

        class Transformer:
            def __call__(
                self,
                hidden_states,
                pooled_projections,
                encoder_hidden_states,
                **kwargs,
            ):
                del kwargs
                calls.append(
                    (
                        pooled_projections.detach().clone(),
                        encoder_hidden_states.detach().clone(),
                    )
                )
                offset = pooled_projections[:, :1, None]
                return (hidden_states + offset,)

        pipeline = StochasticFluxPipeline.__new__(StochasticFluxPipeline)
        pipeline.cfg = SimpleNamespace(mini_batch_size=2)
        pipeline.guidance = None
        pipeline.pipe = SimpleNamespace(transformer=Transformer())
        latents = torch.zeros(3, 2, 2)
        pooled = torch.tensor([[1.0], [2.0], [3.0]])
        prompt = pooled[:, None, :]

        result = pipeline.predict_conditioned(
            latents,
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
            text_ids=torch.zeros(1, 3),
            latent_image_ids=torch.zeros(1, 3),
        )

        torch.testing.assert_close(
            result[:, 0, 0], torch.tensor([1.0, 2.0, 3.0])
        )
        self.assertEqual([call[0].shape[0] for call in calls], [2, 1])


if __name__ == "__main__":
    unittest.main()
