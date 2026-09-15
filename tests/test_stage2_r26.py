"""Final-channel/schema and information-ablation checks for R26 generation."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.r26_generation import parse_value_completion


def test_final_value_is_selected_after_explicit_thinking_channel():
    assert parse_value_completion('<think>{"value":"wrong"}</think>{"value":"right"}', "stop")["value"] == "right"
    assert parse_value_completion('<think>{"value":"wrong"}', "length")["status"] == "truncated"
    assert parse_value_completion('Thinking {"value":"wrong"}\n{"value":"right"}', "stop")["status"] == "parse_error"


@pytest.mark.parametrize("raw", ['{"foo":"x"}', '{"value":1}', '{"value":null}', '{"value":"x","analysis":"y"}', '{"value":"a"}{"value":"b"}'])
def test_generation_rejects_wrong_schema_and_ambiguous_objects(raw):
    assert parse_value_completion(raw, "stop")["status"] == "parse_error"


def test_abstention_is_distinct_from_parse_or_length_failure():
    assert parse_value_completion('{"value":""}', "stop")["status"] == "valid_abstain"
    assert parse_value_completion('{"value":"', "length")["status"] == "truncated"
    assert parse_value_completion('', "stop")["status"] == "parse_error"
    assert parse_value_completion('{"value":"x"}', "length")["status"] == "valid_value"


def test_actual_generation_inputs_include_original_only_in_paired_condition():
    from types import SimpleNamespace
    import torch
    from mmdd_stage2.r26_generation import R26QwenBackend
    from mmdd_stage2.pipeline import LocalizedEvidence
    from mmdd_stage2.data import row_values,CANDIDATE_OPEN,CANDIDATE_CLOSE

    class Processor:
        tokenizer = SimpleNamespace(decode=lambda *a, **k: '{"value":"ok"}')

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            assert kwargs["enable_thinking"] is False
            return {"input_ids": torch.tensor([[1, 2]])}

    backend = R26QwenBackend.__new__(R26QwenBackend)
    backend.device = torch.device("cpu")
    backend.processor = Processor()
    backend.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9), generate=lambda **k: torch.tensor([[1, 2, 8, 9]]))
    backend.generation_context = {}
    backend.generation_records = []
    backend.original_evidence = None
    backend.max_new_tokens = 256
    query = {"columns":[{"column_index":0,"column_name":"name"}],
             "rows":[{"cells":[{"column_index":0,"text":"entity"}]}],
             "hidden_attributes":[{"column_name":"HIDDEN_GT_SENTINEL"}],
             "qrels":"QREL_SENTINEL","query_kind":"KIND_SENTINEL"}
    target = {"columns":[{"column_index":0,"column_name":"color"}],
              "rows":[{"cells":[{"column_index":0,"text":"TARGET_VALUE_SENTINEL"}]}],
              "ground_truth_join":"JOIN_GT_SENTINEL"}
    reader_inputs = []

    def capture_reader(content,**kwargs):
        reader_inputs.append(str(content))
        return {"input_ids":torch.tensor([[101,102]])}

    backend._inputs = capture_reader
    backend.marker_ids = {CANDIDATE_OPEN:101,CANDIDATE_CLOSE:102}
    backend.model.model = lambda **kwargs:SimpleNamespace(last_hidden_state=torch.zeros(1,2,4))
    backend.reader_states(query,target,[])
    assert "TARGET_VALUE_SENTINEL" in reader_inputs[0]
    assert all(value not in reader_inputs[0] for value in ("HIDDEN_GT_SENTINEL","QREL_SENTINEL","KIND_SENTINEL","JOIN_GT_SENTINEL"))
    inputs = {}
    for condition in ("Real-crop", "Real-crop+original", "NoE-fill"):
        backend.condition = condition
        assert backend.generate_value(row_values(query,query["rows"][0]), attribute_name="color",
            evidence=LocalizedEvidence("e", "text", text="CROP_SENTINEL"), original_evidence={"content": "ORIGINAL_SENTINEL"}) == "ok"
        inputs[condition] = str(backend.processor.messages)
    assert "CROP_SENTINEL" in inputs["Real-crop"] and "ORIGINAL_SENTINEL" not in inputs["Real-crop"]
    assert "CROP_SENTINEL" in inputs["Real-crop+original"] and "ORIGINAL_SENTINEL" in inputs["Real-crop+original"]
    assert "CROP_SENTINEL" not in inputs["NoE-fill"] and "ORIGINAL_SENTINEL" not in inputs["NoE-fill"]
    assert all(value not in text for text in inputs.values() for value in
               ("TARGET_VALUE_SENTINEL","HIDDEN_GT_SENTINEL","QREL_SENTINEL","KIND_SENTINEL","JOIN_GT_SENTINEL"))
    assert all(r["generated_tokens"] == 2 and r["finish_reason"] == "stop" for r in backend.generation_records)


def test_stage2_preserves_unattempted_targets_across_entire_c18():
    from run_stage2_r26 import final_candidates
    results = [{"target_id": str(i), "score": -i, "stage2_table_score": -i} for i in range(18)]
    result = final_candidates(results, [], [], {}, {})
    assert [r["target_id"] for r in result] == [str(i) for i in range(18)]
def test_independent_cell_truth_requires_source_row_and_attribute_alignment():
    from audit_stage2_r26_cells import source_cell_truth
    query = {"hidden_attributes":[{"column_name":"Year","source_column_index":1}],
             "rows":[{"row_id":0,"source_row_id":7,"cells":[{"source_column_index":0,"text":"Album A"}]}]}
    source = {"source_table_id":"s","columns":[{"column_index":0,"column_name":"Title"},{"column_index":1,"column_name":"Year"}],
              "rows":[{"row_id":7,"cells":[{"column_index":0,"text":"Album A"},{"column_index":1,"text":"1993"}]}]}
    assert source_cell_truth(query,source,0,"year")["value"] == "1993"
    assert source_cell_truth(query,source,0,"Revenue")["status"] == "unknown"
    query["rows"][0]["cells"][0]["text"] = "Album B"
    assert source_cell_truth(query,source,0,"Year")["reason"] == "visible_source_row_alignment_mismatch"


def test_exact_cell_deletion_matches_production_semantic_recomputation():
    import torch
    from mmdd_stage2.verifier import semantic_joinability
    query = ["1993","other",""]
    targets = ["1993"]
    qemb = torch.tensor([[1.,0.],[.6,.8],[0.,1.]])
    temb = torch.tensor([[1.,0.]])
    before = semantic_joinability(query,targets,query_embeddings=qemb,target_embeddings=temb)
    after = semantic_joinability(["","other",""],targets,query_embeddings=qemb,target_embeddings=temb)
    assert after.coverage == pytest.approx(before.coverage-1/3)
    assert after.mean_similarity == pytest.approx(before.mean_similarity-1/3)


@pytest.mark.parametrize("recover_on_retry",[False,True])
def test_actual_r26_generation_retries_once_and_never_calls_truncation_abstention(recover_on_retry):
    from types import SimpleNamespace
    import torch
    from mmdd_stage2.r26_generation import R26QwenBackend
    from mmdd_stage2.pipeline import LocalizedEvidence

    budgets = []

    def generate(**kwargs):
        budgets.append(kwargs["max_new_tokens"])
        answer = [7,9] if recover_on_retry and len(budgets)==2 else [7]*kwargs["max_new_tokens"]
        return torch.tensor([[1,2,*answer]])

    class Processor:
        tokenizer = SimpleNamespace(decode=lambda answer,**kwargs: '{"value":"ok"}' if int(answer[-1])==9 else '{"value":"')

        def apply_chat_template(self,*args,**kwargs):
            return {"input_ids":torch.tensor([[1,2]])}

    backend = R26QwenBackend.__new__(R26QwenBackend)
    backend.device = torch.device("cpu")
    backend.processor = Processor()
    backend.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9),generate=generate)
    backend.generation_context = {"query_id":"q","target_id":"t","row_id":0}
    backend.generation_records = []
    backend.original_evidence = None
    backend.condition = "Real-crop"
    backend.max_new_tokens = 256
    arguments = {"attribute_name":"year","evidence":LocalizedEvidence("e","text",text="visible evidence")}
    if recover_on_retry:
        assert backend.generate_value({"name":"entity"},**arguments) == "ok"
    else:
        with pytest.raises(ValueError,match="generation_truncated"):
            backend.generate_value({"name":"entity"},**arguments)
    assert budgets == [256,512]
    records = backend.generation_records
    assert [r["attempt"] for r in records] == [1,2]
    assert records[0]["status"] == "truncated" and records[0]["value"] is None
    assert records[0]["generated_tokens"] == 256
    assert records[-1]["status"] == ("valid_value" if recover_on_retry else "truncated")
    assert all(r["status"] != "valid_abstain" for r in records)


def test_actual_r26_driver_resumes_per_generator_and_rejects_changed_inputs(tmp_path,monkeypatch):
    """Stub the expensive models, but execute the real R26 driver/resume/metrics."""
    import shutil
    from types import SimpleNamespace
    import torch
    import run_stage2_r26 as driver
    from run_stage1_r21 import read_rows,write_rows

    root = tmp_path / "repo"
    output = root / "output"
    for name in ("run_stage2_r26.py","mmdd_stage2/r26_generation.py","mmdd_stage1/r26_metrics.py"):
        target = root / "src" / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / "src" / name,target)
    model = root / "hf_models/Qwen3.5-9B"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"synthetic":true}')
    (model / "chat_template.jinja").write_text("synthetic template")
    (output / "stage2/engineering").mkdir(parents=True)
    (output / "stage2/engineering/SUMMARY.json").write_text('{"generation_interface_valid":true}')
    population = [{"query_id":f"q{i}","query_kind":"implicit" if i<32 else "explicit",
                   "positive_target_ids":["t0"]} for i in range(64)]
    write_rows(output / "common/dev_queries.jsonl",population)
    for generator in ("B13","Qwen-Raw"):
        write_rows(output / "stage2/inputs" / generator / "retrieval.jsonl",
            [{"query_id":r["query_id"],"input_pool_sha":generator+"_pool",
              "results":[{"target_id":f"t{i}","score":float(-i),"stage2_table_score":float(-i),"paths":[]} for i in range(18)]}
             for r in population])
    calls = []

    class Verifier:
        def __init__(self,*args,**kwargs):
            pass

        def score_candidates(self,query,*args):
            calls.append(query["table_id"])
            return []

        def verify_direct(self,*args):
            return []

    monkeypatch.setattr(driver,"ROOT",root)
    monkeypatch.setattr(driver,"OUT",output)
    monkeypatch.setattr(driver,"r25_out",lambda root:root / "r25")
    monkeypatch.setattr(driver,"load_candidate_scorer",lambda *a,**k:SimpleNamespace(to=lambda device:None))
    monkeypatch.setattr(driver,"R26QwenBackend",lambda *a,**k:SimpleNamespace(device="cpu",generation_records=[]))
    monkeypatch.setattr(driver,"FeatureStore",SimpleNamespace(from_path=lambda *a,**k:None))
    monkeypatch.setattr(driver,"SimilarityEvidenceRouter",lambda *a,**k:None)
    monkeypatch.setattr(driver,"Stage2Verifier",Verifier)
    monkeypatch.setattr(driver,"load_stage2_index",lambda *a,**k:SimpleNamespace(
        queries={r["query_id"]:{"table_id":r["query_id"]} for r in population},targets={},evidence={}))
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    first = driver.run("B13")
    assert first["rows"] == 192 and len(calls) == 64
    raw = driver.run("Qwen-Raw")
    assert raw["rows"] == 192 and len(calls) == 128
    again = driver.run("B13")
    assert again["rows"] == 192 and len(calls) == 128
    for generator in ("B13","Qwen-Raw"):
        rows = list(read_rows(output / "stage2/pilot" / generator / "results.jsonl"))
        assert {r["generator_id"] for r in rows} == {generator}
        assert len({(r["query_id"],r["condition"],r["input_pool_sha"],r["run_signature"]) for r in rows}) == 192
        assert all(len(r["ranking"]) == 18 for r in rows)
    source = output / "stage2/inputs/B13/retrieval.jsonl"
    source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError,match="Stage2 resume identity changed"):
        driver.run("B13")
    assert len(calls) == 128
