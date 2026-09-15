"""Offline checks of the diagnostic's experimental controls."""
from diagnose_text_prompt_dominance import PLACEHOLDER, bucket, render_inputs


def test_payload_bins_do_not_overlap_at_twenty():
    assert [bucket(n) for n in (0, 2, 3, 5, 6, 10, 11, 20, 21)] == [
        "0-2", "0-2", "3-5", "3-5", "6-10", "6-10", "11-20", "11-20", "21+"]


def test_placeholder_pattern_does_not_count_real_urls_or_ordinary_words():
    assert PLACEHOLDER.findall("https://example.org image url [URL] [image] [N/A]") == ["[URL]", "[image]", "[N/A]"]


class FakeEmbedder:
    def format_model_input(self, *, text, instruction):
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": [{"type": "text", "text": text or "NULL"}]}]

    def _preprocess_inputs(self, conversations):
        return conversations


def test_minimal_keeps_payload_and_chat_roles():
    e = FakeEmbedder()
    original = render_inputs(e, ["abc"], "original", "Long instruction.")[0]
    minimal = render_inputs(e, ["abc"], "minimal", "Long instruction.")[0]
    assert [r["role"] for r in original] == [r["role"] for r in minimal] == ["system", "user"]
    assert original[1] == minimal[1]
    assert minimal[0]["content"][0]["text"] == "Represent this text for retrieval."


def test_empty_payload_control_bypasses_null_fallback():
    e = FakeEmbedder()
    empty = render_inputs(e, [""], "empty", "Long instruction.")[0]
    wrapped = render_inputs(e, [""], "wrapper_empty", "Long instruction.")[0]
    minimal = render_inputs(e, [""], "minimal_empty", "Long instruction.")[0]
    assert empty[0] == wrapped[0]
    assert empty[1]["content"][0]["text"] == ""
    assert wrapped[1]["content"][0]["text"] == "NULL"
    assert minimal[0]["content"][0]["text"] == "Represent this text for retrieval."
    assert minimal[1] == empty[1]
