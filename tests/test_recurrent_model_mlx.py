from __future__ import annotations

import pathlib
import tempfile
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
    from wamrx.recurrent_checkpoint import (
        CheckpointMetadata,
        load_checkpoint,
        save_checkpoint,
        state_hash,
    )
    from wamrx.recurrent_comparison import (
        maximum_feasible_macro_depth,
        promotion_audit,
        run_post_training_manipulations,
    )
    from wamrx.recurrent import ComputeBudget
    from wamrx.recurrent_executor import execute_task
    from wamrx.recurrent_tasks import generate_family, load_split_registry
    from wamrx.recurrent_training import (
        OptimizerConfig,
        build_schedule,
        create_optimizer,
        estimated_schedule_training_flops,
        train_schedule_segment,
        training_tasks,
    )

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

    def test_final_depth_and_adaptive_depths_have_separate_compute_caps(self) -> None:
        registered = [1, 2, 3, 4, 6, 8, 12]
        budget = ComputeBudget(
            maximum_inference_flops=self.config.maximum_inference_flops,
            maximum_macro_steps=12,
            maximum_micro_steps_per_macro=4,
            maximum_total_micro_steps=48,
            maximum_retrieval_calls=4,
            maximum_tool_calls=4,
        )
        for arm_id in ARM_IDS:
            model = RecurrentReasoner(arm_id, self.config)
            final_depth = maximum_feasible_macro_depth(
                model,
                registered,
                micro_steps=1,
                readout_every_position=False,
            )
            adaptive_depth = maximum_feasible_macro_depth(
                model,
                registered,
                micro_steps=1,
                readout_every_position=True,
            )
            output, trace, compute = execute_task(
                model,
                self.codec,
                self.tasks[0],
                budget=budget,
                maximum_answer_instability=0.02,
                maximum_macro_steps=final_depth,
                execution_policy="final_depth",
            )
            output.validate(trace)
            compute.validate(budget)
            if arm_id == "fixed-depth-v1":
                self.assertEqual((final_depth, adaptive_depth), (4, 4))
            else:
                self.assertEqual(final_depth, 12)
                self.assertEqual(adaptive_depth, 6)

    def test_post_training_manipulation_paths_are_executable(self) -> None:
        optimizer_config = OptimizerConfig.load(
            ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
        )
        registry = load_split_registry(
            ROOT / "contracts" / "wamrx_recurrent_splits.json"
        )
        tasks = training_tasks(registry["splits"]["train"])
        schedule = build_schedule(
            tasks, optimizer_config, seed=41001, updates=8, batch_size=2
        )
        model = RecurrentReasoner("flat-recurrent-v1", self.config)
        checks = run_post_training_manipulations(
            model, self.codec, self.tasks[0], schedule=schedule
        )
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(checks.values()))

    def test_training_compute_distinguishes_schedule_realization(self) -> None:
        optimizer_config = OptimizerConfig.load(
            ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
        )
        registry = load_split_registry(
            ROOT / "contracts" / "wamrx_recurrent_splits.json"
        )
        tasks = training_tasks(registry["splits"]["train"])
        schedule = build_schedule(
            tasks, optimizer_config, seed=41001, updates=2, batch_size=2
        )
        totals = {}
        for arm_id in ARM_IDS:
            model = RecurrentReasoner(arm_id, self.config)
            totals[arm_id] = estimated_schedule_training_flops(model, schedule)
        self.assertEqual(len(set(totals.values())), 3)
        self.assertTrue(all(value > 0 for value in totals.values()))

    def test_checkpoint_restores_model_and_optimizer_state(self) -> None:
        optimizer_config = OptimizerConfig.load(
            ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
        )
        registry = load_split_registry(
            ROOT / "contracts" / "wamrx_recurrent_splits.json"
        )
        tasks = training_tasks(registry["splits"]["train"])
        schedule = build_schedule(
            tasks, optimizer_config, seed=41001, updates=2, batch_size=2
        )
        mx.random.seed(41001)
        model = RecurrentReasoner("flat-recurrent-v1", self.config)
        optimizer = create_optimizer(optimizer_config, total_updates=2)
        report = train_schedule_segment(
            model,
            self.codec,
            tasks,
            schedule,
            optimizer_config,
            optimizer=optimizer,
            start_update=0,
            stop_update=1,
        )
        metadata = CheckpointMetadata(
            run_id="checkpoint-test",
            arm_id=model.arm_id,
            seed=41001,
            completed_updates=1,
            schedule_hash=report["schedule_hash"],
            model_implementation_hash="a" * 64,
            training_config_hash="b" * 64,
            split_registry_hash="c" * 64,
            realized_training_flops=report["realized_training_flops"],
            training_examples_seen=report["examples_seen"],
        )
        before = state_hash(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "checkpoint.safetensors"
            saved = save_checkpoint(path, model, optimizer, metadata)
            mx.random.seed(999)
            restored_model = RecurrentReasoner("flat-recurrent-v1", self.config)
            restored_optimizer = create_optimizer(
                optimizer_config, total_updates=2
            )
            loaded = load_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                expected=metadata,
            )
        self.assertEqual(before, saved["state_hash"])
        self.assertEqual(before, loaded["state_hash"])
        self.assertEqual(saved["file_sha256"], loaded["file_sha256"])

    def test_promotion_audit_consumes_every_frozen_analysis_family(self) -> None:
        seeds = (41001, 41002, 41003, 41004, 41005)
        depths = (1, 2, 3, 4, 6, 8, 12)
        primary = {
            "fixed-depth-v1": 4,
            "flat-recurrent-v1": 12,
            "hierarchical-recurrent-v1": 12,
        }
        adaptive = {
            "fixed-depth-v1": 4,
            "flat-recurrent-v1": 6,
            "hierarchical-recurrent-v1": 6,
        }
        base_accuracy = {
            "fixed-depth-v1": 0.5,
            "flat-recurrent-v1": 0.6,
            "hierarchical-recurrent-v1": 0.7,
        }
        regions = {
            "algorithmic": ("algorithmic",),
            "structured": ("structured",),
            "multiview": ("finance", "operations"),
        }
        rows = []
        for arm_id in ARM_IDS:
            for seed in seeds:
                for split in ("id", "ood"):
                    for depth in depths:
                        for family in ("algorithmic", "structured", "multiview"):
                            common = {
                                "arm_id": arm_id,
                                "seed": seed,
                                "split": split,
                                "mode": "final_depth",
                                "requested_macro_depth": depth,
                                "correct": int(base_accuracy[arm_id] * 100),
                                "examples": 100,
                                "accuracy": base_accuracy[arm_id],
                                "total_inference_flops": 100,
                                "median_inference_flops": (
                                    100
                                    if arm_id == "fixed-depth-v1"
                                    else 80 + depth * 5
                                ),
                            }
                            rows.append(
                                {**common, "grouping": "family", "label": family}
                            )
                            for region in regions[family]:
                                rows.append(
                                    {**common, "grouping": "region", "label": region}
                                )
                    for mode, flops in (("maximum_depth", 100), ("adaptive", 70)):
                        for family in ("algorithmic", "structured", "multiview"):
                            rows.append(
                                {
                                    "arm_id": arm_id,
                                    "seed": seed,
                                    "split": split,
                                    "mode": mode,
                                    "requested_macro_depth": adaptive[arm_id],
                                    "grouping": "family",
                                    "label": family,
                                    "correct": int(base_accuracy[arm_id] * 100),
                                    "examples": 100,
                                    "accuracy": base_accuracy[arm_id],
                                    "total_inference_flops": flops,
                                    "median_inference_flops": flops,
                                }
                            )
        audit = promotion_audit(
            rows,
            seeds=seeds,
            primary_depths=primary,
            adaptive_depths=adaptive,
            manipulations={f"M{index}": True for index in range(1, 9)},
            training_compute_control={
                "ood_tests": {
                    "flat-recurrent-v1": {"passes": True},
                    "hierarchical-recurrent-v1": {"passes": True},
                }
            },
        )
        self.assertEqual(audit["decision"], "COMPLETE_ADOPT_HIERARCHY")


if __name__ == "__main__":
    unittest.main()
