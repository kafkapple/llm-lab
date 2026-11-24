from pathlib import Path

import pytest

from qai_poc.eval_utils import bootstrap_ci, canonical_hash, sha256_file, token_f1


def test_longbench_qa_f1_normalization() -> None:
    assert token_f1("The Eiffel Tower.", "Eiffel Tower") == 1.0
    assert token_f1("red blue", "red green") == 0.5
    assert token_f1("", "") == 1.0


def test_bootstrap_is_deterministic_and_rejects_empty() -> None:
    assert bootstrap_ci([0.0, 1.0], 100, 7) == bootstrap_ci(
        [0.0, 1.0], 100, 7
    )
    with pytest.raises(ValueError):
        bootstrap_ci([], 100, 7)


def test_hashes_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_text("abc")
    assert sha256_file(path).startswith("ba7816bf")
    assert canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
