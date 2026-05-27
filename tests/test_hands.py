"""Tests for poker hand evaluation."""

from __future__ import annotations

import pytest

from poker_yolo.hands import (
    HandValidationError,
    evaluate_hand,
    validate_hand_cards,
)


def test_two_pair() -> None:
    result = evaluate_hand(["10S", "10D", "3D", "3C", "8S"])
    assert not isinstance(result, HandValidationError)
    assert result.hand_type == "two_pair"
    assert "десятки" in result.description_ru


def test_royal_flush() -> None:
    result = evaluate_hand(["AS", "KS", "QS", "JS", "10S"])
    assert result.hand_type == "royal_flush"


def test_full_house() -> None:
    result = evaluate_hand(["KC", "KD", "KH", "2S", "2C"])
    assert result.hand_type == "full_house"


def test_wheel_straight() -> None:
    result = evaluate_hand(["AS", "2C", "3D", "4H", "5S"])
    assert result.hand_type == "straight"
    assert "пятки" in result.description_ru


def test_rejects_wrong_card_count() -> None:
    error = validate_hand_cards(["AH", "KD", "QC"])
    assert error is not None
    assert "5" in error.message_ru


def test_rejects_duplicate_cards() -> None:
    result = evaluate_hand(["AH", "AH", "KD", "QC", "JS"])
    assert isinstance(result, HandValidationError)
    assert "повторяются" in result.message_ru


def test_high_card() -> None:
    result = evaluate_hand(["2C", "5D", "7H", "9S", "JC"])
    assert result.hand_type == "high_card"
