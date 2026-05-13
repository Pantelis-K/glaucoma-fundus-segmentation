"""
Tests for input validation and ID resolution in src/inference.py.
Covers out-of-range integers, decimal strings, and non-numeric strings.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference import _parse_id, _resolve_image


# ---------------------------------------------------------------------------
# _resolve_image: out-of-range and boundary IDs
# ---------------------------------------------------------------------------

def test_resolve_image_above_range():
    # 651 is outside ORIGA (1-650) so no file exists
    img_path, padded_id = _resolve_image("651")
    assert img_path is None
    assert padded_id == "651"


def test_resolve_image_zero():
    img_path, padded_id = _resolve_image("0")
    assert img_path is None
    assert padded_id == "000"


def test_resolve_image_large_value():
    img_path, padded_id = _resolve_image("9999")
    assert img_path is None
    assert padded_id == "9999"


def test_resolve_image_pads_single_digit():
    _, padded_id = _resolve_image("1")
    assert padded_id == "001"


def test_resolve_image_pads_two_digits():
    _, padded_id = _resolve_image("42")
    assert padded_id == "042"


# ---------------------------------------------------------------------------
# _parse_id: rejects decimals, strings, and empty input
# ---------------------------------------------------------------------------

def test_parse_id_rejects_decimal():
    # "3.14".isdigit() is False — should re-prompt, then accept a valid ID
    with patch("builtins.input", side_effect=["3.14", "586"]):
        result = _parse_id()
    assert result == "586"


def test_parse_id_rejects_string():
    with patch("builtins.input", side_effect=["hello", "001"]):
        result = _parse_id()
    assert result == "001"


def test_parse_id_rejects_empty_string():
    with patch("builtins.input", side_effect=["", "200"]):
        result = _parse_id()
    assert result == "200"


def test_parse_id_rejects_multiple_bad_inputs_in_sequence():
    # Several invalid inputs before a valid one
    with patch("builtins.input", side_effect=["abc", "3.14", "-5", "42"]):
        result = _parse_id()
    assert result == "42"


def test_parse_id_accepts_valid_on_first_attempt():
    with patch("builtins.input", return_value="586"):
        result = _parse_id()
    assert result == "586"
