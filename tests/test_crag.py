import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.crag import _extract_json, has_usable, select_context


def test_extract_json_from_noisy_llm_output():
    noisy = 'Sure! Here is my assessment: {"verdict": "correct", "reason": "directly answers"} hope that helps'
    assert _extract_json(noisy) == '{"verdict": "correct", "reason": "directly answers"}'


def test_extract_json_passthrough_without_braces():
    assert _extract_json("no json here") == "no json here"


def test_has_usable_requires_correct():
    docs = [{"grade": "ambiguous"}, {"grade": "incorrect"}]
    assert has_usable(docs) is False
    assert has_usable([{"grade": "correct"}]) is True


def test_select_context_prefers_correct_then_ambiguous():
    correct = {"grade": "correct", "content": "c"}
    ambiguous = {"grade": "ambiguous", "content": "a"}
    incorrect = {"grade": "incorrect", "content": "i"}
    assert select_context([correct, incorrect]) == [correct]
    assert select_context([ambiguous, incorrect]) == [ambiguous]
    assert select_context([incorrect]) == [incorrect]
