from __future__ import annotations

import pathlib
import unittest

try:
    import mlx.core as mx

    from wamrx.recurrent_model import (
        ARM_IDS,
        ByteCodec,
        RecurrentModelConfig,
        RecurrentModelError,
        RecurrentReasoner,
        batch_arrays,
        decode_batch,
        parameter_count,
        token_loss,
    )
    from wamrx.recurrent import ComputeBudget
    from wamrx.recurrent_executor import execute_task
    from wamrx.recurrent_tasks import generate_family, load_split_registry

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless(MLX_AVAILABLE, "MLX runtime is optional")
class RecurrentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RecurrentModelConfig.load(
            ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
        )
        cls.codec = ByteCodec(
            maximum_input_tokens=cls.config.maximum_input_tokens,
            maximum_output_tokens=cls.config.maximum_output_tokens,
        )
        registry = load_split_registry(
            ROOT / "contracts" / "wamrx_recurrent_splits.json"
        )
        cls.tasks = list(
            generate_family("train", "algorithmic", registry["splits"]["train"])[
                :2
            ]
        )

    def test_all_arms_hit_exact_parameter_targets(self) -> None:
        for arm_id in ARM_IDS:
            model = RecurrentReasoner(arm_id, self.config)
            self.assertEqual(
                parameter_count(model), self.config.parameter_targets[arm_id]
            )

    def test_all_arms_forward_with_shared_shapes(self) -> None:
        tokens, targets, target_mask = batch_arrays(self.codec, self.tasks)
        for arm_id in ARM_IDS:
            model = RecurrentReasoner(arm_id, self.config)
            outputs = model(tokens, macro_steps=2, micro_steps=1)
            mx.eval(outputs["logits"])
            self.assertEqual(
                outputs["logits"].shape,
                (2, self.config.maximum_output_tokens, self.config.vocabulary_size),
            )
            loss = token_loss(
                outputs,
                targets,
                target_mask,
                intermediate_weight=0.25,
            )
            mx.eval(loss)
            self.assertGreater(float(loss), 0.0)
            self.assertEqual(len(decode_batch(self.codec, outputs["logits"])), 2)

    def test_registered_depths_respect_common_flop_budget_or_fail_closed(self) -> None:
        for arm_id in ARM_IDS:
            model = RecurrentReasoner(arm_id, self.config)
            estimate = model.estimated_inference_flops(
                input_tokens=self.config.maximum_input_tokens,
                macro_steps=12,
                micro_steps=4,
            )
            if arm_id == "hierarchical-recurrent-v1":
                self.assertGreater(estimate, self.config.maximum_inference_flops)
            else:
                self.assertLessEqual(estimate, self.config.maximum_inference_flops)

    def test_codec_rejects_silent_truncation(self) -> None:
        with self.assertRaises(RecurrentModelError):
            self.codec.encode_problem({"too_long": "x" * 2000})

    def test_untrained_execution_still_emits_valid_fail_closed_trace(self) -> None:
        model = RecurrentReasoner("flat-recurrent-v1", self.config)
        budget = ComputeBudget(
            maximum_inference_flops=self.config.maximum_inference_flops,
            maximum_macro_steps=12,
            maximum_micro_steps_per_macro=4,
            maximum_total_micro_steps=48,
            maximum_retrieval_calls=4,
            maximum_tool_calls=4,
        )
        output, trace, compute = execute_task(
            model,
            self.codec,
            self.tasks[0],
            budget=budget,
            maximum_answer_instability=0.02,
            maximum_macro_steps=2,
        )
        trace.validate()
        output.validate(trace)
        compute.validate(budget)
        self.assertIn(
            output.halt_reason,
            {"resolved", "resolved_at_budget", "unresolved_budget"},
        )


if __name__ == "__main__":
    unittest.main()
