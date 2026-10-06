"""Controlled evidence and text-span experiments, without downloaded models or GPUs."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mmdd_stage2 import experiments, recovery, text_span
from mmdd_stage2.common import write_json, write_jsonl
from mmdd_stage2.localizer import ATTR_CLOSE, ATTR_OPEN, ROW_CLOSE, ROW_OPEN


class CharacterTokenizer:
    all_special_tokens = [ATTR_OPEN, ATTR_CLOSE, ROW_OPEN, ROW_CLOSE]

    def convert_tokens_to_ids(self, token):
        return 1000 + self.all_special_tokens.index(token)

    def __call__(self, text, return_tensors=None, **kwargs):
        matches = list(re.finditer(r"<\|[^>]+\|>|[\s\S]", text))
        ids = [self.convert_tokens_to_ids(m.group()) if m.group() in self.all_special_tokens else ord(m.group())
               for m in matches]
        result = {"input_ids": ids, "offset_mapping": [(m.start(), m.end()) for m in matches]}
        if return_tensors:
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def selector(mode, model=None):
    return text_span.TextSpanSelector(SimpleNamespace(tokenizer=CharacterTokenizer()), model,
                                     {"mode": mode, "span_tokens": 8, "window_tokens": 20, "overlap_tokens": 4,
                                      "layers": [0]})


ROW = {"query_row_id": 0, "cells": [{"column_name": "Name", "text": "Alice"}]}


def test_windows_cover_tail_without_an_extra_duplicate_window():
    assert text_span.token_windows(0, 10, 2) == []
    assert text_span.token_windows(18, 10, 2) == [(0, 10), (8, 18)]
    assert text_span.token_windows(19, 10, 2) == [(0, 10), (8, 18), (16, 19)]
    with pytest.raises(ValueError):
        text_span.token_windows(10, 10, 10)


def test_peak_span_handles_edges_and_ties():
    assert text_span.peak_span(np.array([0., 1., 9., 8., 0.]), 3) == (1, 4)
    assert text_span.peak_span(np.zeros(8), 3) == (0, 3)
    assert text_span.peak_span(np.array([1., 2.]), 10) == (0, 2)


def test_joint_relevance_requires_both_conditions():
    # Evidence 0 has only entity support; evidence 1 has both; evidence 2 only attribute.
    values = torch.tensor([[1., 0.], [.7, .7], [0., 1.], [1., 0.], [0., 1.]])
    joint = text_span.conditional_relevance([values], 3, 1)
    assert np.argmax(joint) == 1
    assert np.argmax(text_span.conditional_relevance([values], 3, 1, "entity")) == 0
    assert np.argmax(text_span.conditional_relevance([values], 3, 1, "attribute")) == 2


def test_lexical_selects_condition_words_and_caches_without_counting_cost_twice():
    s = selector("lexical")
    content = "xxxxxxxxxxxxxxxxxxxx Alice born 1980 yyyyyyyyyyyyyyyy"
    result = s.select(ROW, "born", content)
    assert result["start_char"] > 0 and "Alice" in result["text"]
    assert result["text"] == content[result["start_char"]:result["end_char"]]
    assert result["selected_tokens"] <= 8 and result["forwards"] == 0
    cached = s.select(ROW, "born", content)
    assert cached["cache_hit"] and cached["seconds"] == 0


def test_short_text_never_runs_a_localizer_forward():
    assert selector("joint").select(ROW, "born", "1980")["reason"] == "SHORT_TEXT"


def test_recovery_message_preserves_prefix_default_and_records_optional_span():
    generator = recovery.Generator.__new__(recovery.Generator)
    generator.c = {"text_chars": 12}
    generator.crops, generator.text_spans, generator.text_selector = {}, {}, None
    task = {"task_id": "x", "row": ROW, "column_name": "born", "evidence_ids": ["e"]}
    query = {"assets": {"e": {"asset_type": "text", "content": "Alice born 1980. More text"}}}
    message = generator.message(task, query)
    assert message[0]["content"][1]["text"] == "\nE1 (text):\nAlice born 1\n"
    seen = []
    def pick(row, attribute, text):
        seen.append(text)
        return {"text": "born", "start_char": 6, "end_char": 10}
    generator.text_selector = SimpleNamespace(select=pick)
    message = generator.message(task, query)
    assert seen == ["Alice born 1"]  # cannot read beyond the baseline prefix
    assert message[0]["content"][1]["text"] == "\nE1 (text):\nborn\n"
    assert generator.text_spans["x"][0]["evidence_id"] == "e"


def test_recovery_counts_tokens_once_for_shared_requests():
    class FakeGenerator:
        crops, text_spans, text_selector = {}, {}, None
        def run(self, tasks, query):
            return {t["task_id"]: {"input_limit": False, "length_limit": False, "raw_completion": '"1980"',
                                   "prompt_tokens": 10, "generated_tokens": 3} for t in tasks}
    query = {"query_id": "q", "rows": [{**ROW, "query_row_id": i} for i in range(5)],
             "tables": {"t": {"columns": [{"column_id": 0, "column_name": "Year"}]}},
             "assets": {"e": {"asset_type": "text", "content": "Alice was born in 1980"}}}
    view = {"attribute": "year", "column_name": "Year", "evidence_ids": ["e"],
            "donor_links": [{"target_id": "t", "column_id": 0}]}
    result = recovery.recover_query(FakeGenerator(), query, [view, view], 3000)
    assert result["model_inputs"] == 5
    assert result["generation_tokens"] == {"prompt_tokens": 50, "generated_tokens": 15}
    assert len(result["tasks"]) == 10 and result["text_localization"]["forwards"] == 0


def test_model_span_maps_window_offsets_back_to_original_prefix(monkeypatch):
    s = selector("joint")
    def scores(row, attribute, text):
        return np.array([float(c == "Z") for c in text]), [(i, i + 1) for i in range(len(text))]
    monkeypatch.setattr(s, "_model_scores", scores)
    text = "a" * 25 + "ZZZZ" + "b" * 20
    result = s.select(ROW, "born", text)
    assert "ZZZZ" in result["text"]
    assert result["text"] == text[result["start_char"]:result["end_char"]]
    assert result["selected_tokens"] == 8


class FakeInner(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.language_model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(v_proj=self.layer))])
        self.rope_deltas = "original"
        self.fail = fail

    def forward(self, input_ids, **kwargs):
        self.rope_deltas = "mutated"
        x = input_ids.float().unsqueeze(-1)
        self.layer(torch.cat([x.sin(), x.cos(), (x / 2).sin(), (x / 2).cos()], dim=-1))
        if self.fail:
            raise RuntimeError("synthetic forward failure")


@pytest.mark.parametrize("fail", [False, True])
def test_value_hooks_capture_conditions_and_cleanup_even_on_failure(fail):
    model = torch.nn.Module()
    model.model = FakeInner(fail)
    s = selector("joint", model)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic"):
            s._model_scores(ROW, "birth year", "Alice born 1980")
    else:
        scores, offsets = s._model_scores(ROW, "birth year", "Alice born 1980")
        assert len(scores) == len("Alice born 1980") == len(offsets)
        assert np.isfinite(scores).all() and s.forwards == 1
    assert not model.model.layer._forward_hooks
    assert model.model.rope_deltas == "original"


def test_teacher_and_cosine_use_same_bag_and_modality_counts():
    view = {"evidence_ids": ["a", "b", "i"], "donor_links": [{"target_id": "t"}]}
    paths = {"t": {"a": {"modality": "text", "residual": 2}, "b": {"modality": "text", "residual": 1},
                   "i": {"modality": "image", "residual": 0}}}
    cosine = SimpleNamespace(score=lambda q, e: {"a": .1, "b": .9, "i": .2}[e])
    assert experiments.choose_evidence(view, paths, "q", "teacher")[0] == ["a", "i"]
    assert experiments.choose_evidence(view, paths, "q", "cosine", cosine)[0] == ["b", "i"]
    assert experiments.choose_evidence({**view, "evidence_ids": []}, {}, "q", "teacher")[0] == []


def test_group_sampling_is_nested_and_never_splits_a_source():
    population = [{"query_id": str(i), "source_group": str(i // 2)} for i in range(20)]
    small = experiments.select_population(population, 2)
    bigger = experiments.select_population(population, 3)
    assert len(small) == 4 and all(r in bigger for r in small)
    assert experiments.select_population(population, 0) == population


def test_paired_comparison_rejects_dropped_queries_and_changed_groups():
    row = {"query_id": "q", "split": "dev", "kind": "implicit", "source_group": "g", "R10": .5}
    with pytest.raises(ValueError, match="identical"):
        experiments.paired_rows([row], [], "a", "b", ["R10"])
    with pytest.raises(ValueError, match="metadata"):
        experiments.paired_rows([row], [{**row, "source_group": "other"}], "a", "b", ["R10"])
    result = experiments.paired_rows([row], [{**row, "R10": 0}], "a", "b", ["R10"], replicates=10)
    assert result[0]["delta"] == .5 and result[0]["W"] == 1


def test_prepare_keeps_source_and_baseline_results_immutable(tmp_path):
    source = tmp_path / "source"
    write_json(source / "config.json", {"paths": {"run_root": str(source)}})
    (source / "catalog.sqlite").write_bytes(b"synthetic read-only catalog")
    for split in ("dev", "test"):
        write_json(source / "population" / f"{split}.json", [{"query_id": split, "source_group": "g", "split": split}])
        write_jsonl(source / "plans" / f"{split}.jsonl", [{"query_id": split, "views": []}])
        write_json(source / "recovery" / split / f"{split}.json", {"query_id": split, "bridges": []})
    before = (source / "config.json").read_bytes()
    output = tmp_path / "arm"
    experiments.prepare(source, output, "baseline", None, None, 0, ["dev"])
    assert (source / "config.json").read_bytes() == before
    assert json.loads((output / "population/test.json").read_text()) == []
    assert (output / "catalog.sqlite").resolve() == source / "catalog.sqlite"
    assert not (output / "recovery/dev/dev.json").is_symlink()
    experiments.verify(output)
    (output / "plans/dev.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="frozen_files changed"):
        experiments.verify(output)
    with pytest.raises(ValueError, match="fresh"):
        experiments.prepare(source, output, "baseline", None, None, 0, ["dev"])
