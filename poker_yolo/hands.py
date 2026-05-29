"""Poker hand evaluation from detected playing cards."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

RANK_VALUES: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}

RANK_NAMES_RU: dict[int, str] = {
    2: "двойки",
    3: "тройки",
    4: "четвёрки",
    5: "пятки",
    6: "шестёрки",
    7: "семёрки",
    8: "восьмёрки",
    9: "девятки",
    10: "десятки",
    11: "валеты",
    12: "дамы",
    13: "короли",
    14: "тузы",
}

HAND_NAMES_RU: dict[str, str] = {
    "high_card": "Старшая карта",
    "pair": "Пара",
    "two_pair": "Две пары",
    "three_of_a_kind": "Сет",
    "straight": "Стрит",
    "flush": "Флеш",
    "full_house": "Фул-хаус",
    "four_of_a_kind": "Каре",
    "straight_flush": "Стрит-флеш",
    "royal_flush": "Роял-флеш",
}

# Canonical order for Grafana bar chart / confusion matrix (10 combo classes).
COMBO_CLASSES: tuple[str, ...] = tuple(HAND_NAMES_RU.keys())


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str
    label: str

    @classmethod
    def parse(cls, label: str) -> Card:
        text = label.strip().upper()
        if len(text) < 2:
            raise ValueError(f"Invalid card label: {label!r}")
        suit = text[-1]
        rank_token = text[:-1]
        if suit not in SUIT_SYMBOLS or rank_token not in RANK_VALUES:
            raise ValueError(f"Invalid card label: {label!r}")
        return cls(rank=RANK_VALUES[rank_token], suit=suit, label=text)

    def display(self) -> str:
        rank_token = next(k for k, v in RANK_VALUES.items() if v == self.rank)
        return f"{rank_token}{SUIT_SYMBOLS[self.suit]}"


@dataclass(frozen=True)
class HandEvaluation:
    hand_type: str
    cards: tuple[Card, ...]
    description_ru: str

    @property
    def name_ru(self) -> str:
        return HAND_NAMES_RU[self.hand_type]


@dataclass(frozen=True)
class HandValidationError:
    message_ru: str
    detected_labels: tuple[str, ...]


def parse_card_label(label: str) -> Card:
    return Card.parse(label)


def validate_hand_cards(labels: Iterable[str]) -> HandValidationError | None:
    cards: list[Card] = []
    for label in labels:
        try:
            cards.append(parse_card_label(label))
        except ValueError:
            return HandValidationError(
                message_ru=f"Неизвестная карта: {label}",
                detected_labels=tuple(labels),
            )

    if len(cards) != 5:
        return HandValidationError(
            message_ru=f"Обнаружено {len(cards)} карт, для покерной комбинации нужно ровно 5",
            detected_labels=tuple(c.label for c in cards),
        )

    if len({c.label for c in cards}) != 5:
        return HandValidationError(
            message_ru="На изображении повторяются карты — такой комбинации не существует",
            detected_labels=tuple(c.label for c in cards),
        )

    return None


def evaluate_hand(labels: Iterable[str]) -> HandEvaluation | HandValidationError:
    error = validate_hand_cards(labels)
    if error is not None:
        return error

    cards = tuple(parse_card_label(label) for label in labels)
    return _evaluate_five_cards(cards)


def _evaluate_five_cards(cards: tuple[Card, ...]) -> HandEvaluation:
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high == 14 and ranks == [14, 13, 12, 11, 10]:
        return HandEvaluation("royal_flush", cards, "Роял-флеш")

    if is_flush and straight_high is not None:
        return HandEvaluation(
            "straight_flush",
            cards,
            f"Стрит-флеш, старшая карта — {_rank_name_ru(straight_high)}",
        )

    if counts == [4, 1]:
        quad_rank = _rank_by_count(rank_counts, 4)
        return HandEvaluation("four_of_a_kind", cards, f"Каре — { _rank_name_ru(quad_rank)}")

    if counts == [3, 2]:
        trip_rank = _rank_by_count(rank_counts, 3)
        pair_rank = _rank_by_count(rank_counts, 2)
        return HandEvaluation(
            "full_house",
            cards,
            f"Фул-хаус — { _rank_name_ru(trip_rank)} и {_rank_name_ru(pair_rank)}",
        )

    if is_flush:
        return HandEvaluation(
            "flush",
            cards,
            f"Флеш, старшая карта — {_rank_name_ru(ranks[0])}",
        )

    if straight_high is not None:
        return HandEvaluation(
            "straight",
            cards,
            f"Стрит, старшая карта — {_rank_name_ru(straight_high)}",
        )

    if counts == [3, 1, 1]:
        trip_rank = _rank_by_count(rank_counts, 3)
        return HandEvaluation("three_of_a_kind", cards, f"Сет — {_rank_name_ru(trip_rank)}")

    if counts == [2, 2, 1]:
        pairs = sorted((r for r, c in rank_counts.items() if c == 2), reverse=True)
        return HandEvaluation(
            "two_pair",
            cards,
            f"Две пары — {_rank_name_ru(pairs[0])} и {_rank_name_ru(pairs[1])}",
        )

    if counts == [2, 1, 1, 1]:
        pair_rank = _rank_by_count(rank_counts, 2)
        return HandEvaluation("pair", cards, f"Пара — {_rank_name_ru(pair_rank)}")

    return HandEvaluation(
        "high_card",
        cards,
        f"Старшая карта — {_rank_name_ru(ranks[0])}",
    )


def _straight_high(ranks: list[int]) -> int | None:
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return None
    if unique[0] + 4 == unique[-1]:
        return unique[-1]
    if unique == [2, 3, 4, 5, 14]:
        return 5
    return None


def _rank_by_count(rank_counts: Counter[int], count: int) -> int:
    return max(rank for rank, qty in rank_counts.items() if qty == count)


def _rank_name_ru(rank: int) -> str:
    return RANK_NAMES_RU[rank]
