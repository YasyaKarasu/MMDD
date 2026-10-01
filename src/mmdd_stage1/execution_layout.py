"""Execution-only amendment. No model, loss, data or optimizer change."""
TEACHER_CHUNK_LADDER = (256, 128, 64, 32, 16, 8, 4, 2, 1)
TEACHER_INITIAL_CHUNK = 256
TEACHER_INFERENCE_CHUNK = 256
TEACHER_LAYOUT_REVISION = "v4_1_speed_c_full_list_20260926"


def teacher_numerical_layout() -> dict:
    return {
        "logical_batch": 8,
        "candidate_chunk_ladder": list(TEACHER_CHUNK_LADDER),
        "teacher_scorer": "single_graph_first_query_cache_v1",
        "fallback": "same_batch_same_chunk_two_pass_then_halve",
        "retry_scope": "whole_logical_batch_before_optimizer_step",
        "initial_mode_each_batch": "single_graph",
        "inference_candidate_chunk": TEACHER_INFERENCE_CHUNK,
        "numerical_layout_revision": TEACHER_LAYOUT_REVISION,
    }
