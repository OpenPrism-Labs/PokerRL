"""Private, deterministic six-seat no-limit Hold'em over PokerKit.

Decks deal clockwise from the small blind, one card per seat for two rounds,
then burn/flop/burn/turn/burn/river. Chip arrays always use external seat order.
"""

from dataclasses import dataclass
from random import Random
from typing import Final, Literal, assert_never

from pokerkit import Automation, Card, NoLimitTexasHoldem, State

ActionKind = Literal["fold", "check_call", "raise_to"]
_CARDS: Final = tuple(Card.parse(" ".join(r + s for r in "23456789TJQKA" for s in "cdhs")))
_AUTOMATIONS: Final = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    amount: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("fold", "check_call", "raise_to") or type(self.amount) is not int:
            raise ValueError("Action requires a known kind and integer amount.")


@dataclass(frozen=True, slots=True)
class LegalActions:
    can_fold: bool
    can_check_call: bool
    call_amount: int
    min_raise_to: int | None
    max_raise_to: int | None


@dataclass(frozen=True, slots=True)
class PublicAction:
    seat: int
    street: int
    kind: ActionKind
    paid: int
    raise_to: int | None
    faced_bet: bool


@dataclass(frozen=True, slots=True)
class Observation:
    seat: int
    button: int
    street: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    stacks: tuple[int, ...]
    bets: tuple[int, ...]
    contributions: tuple[int, ...]
    active: tuple[bool, ...]
    all_in: tuple[bool, ...]
    pot: int
    big_blind: int
    legal: LegalActions
    actions: tuple[PublicAction, ...]


@dataclass(frozen=True, slots=True)
class HandResult:
    button: int
    board: tuple[int, ...]
    payoffs: tuple[int, ...]
    actions: tuple[PublicAction, ...]


def _check_stacks(stacks: tuple[int, ...]) -> None:
    if len(stacks) != 6 or any(type(n) is not int or n <= 0 for n in stacks):
        raise ValueError("Exactly six positive integer stacks are required.")


def _check_button(button: int) -> None:
    if type(button) is not int or button not in range(6):
        raise ValueError("Button must be an integer seat from 0 to 5.")


class HoldemEnv:
    """Mutable hand driver; policies receive only immutable public observations."""

    def __init__(
        self,
        seed: int,
        *,
        stacks: tuple[int, ...] = (200,) * 6,
        small_blind: int = 1,
        big_blind: int = 2,
        button: int = 5,
    ) -> None:
        _check_stacks(stacks)
        _check_button(button)
        if type(seed) is not int:
            raise ValueError("Seed must be an integer.")
        if (
            type(small_blind) is not int
            or type(big_blind) is not int
            or not 0 < small_blind <= big_blind
        ):
            raise ValueError("Blinds must be positive integers with small <= big.")
        self._rng = Random(seed)
        self._defaults = stacks
        self._small_blind = small_blind
        self._big_blind = big_blind
        self._button = button
        self._state: State | None = None
        self._seats: tuple[int, ...] = ()
        self._indices: tuple[int, ...] = ()
        self._holes: list[list[int]] = []
        self._board: list[int] = []
        self._actions: list[PublicAction] = []
        self._deck: tuple[int, ...] = ()
        self._cursor = 0

    @property
    def actor(self) -> int | None:
        state = self._state
        if state is None or state.actor_index is None:
            return None
        return self._seats[state.actor_index]

    @property
    def terminal(self) -> bool:
        return self._state is None or not self._state.status

    def reset(
        self,
        *,
        button: int | None = None,
        stacks: tuple[int, ...] | None = None,
        deck: tuple[int, ...] | None = None,
    ) -> None:
        """Start a rebought hand; a supplied deck does not advance the local RNG."""
        if not self.terminal:
            raise RuntimeError("Cannot reset an unfinished hand.")
        next_button = (
            (self._button if self._state is None else (self._button + 1) % 6)
            if button is None
            else button
        )
        starting = self._defaults if stacks is None else stacks
        _check_button(next_button)
        _check_stacks(starting)
        if deck is not None and (
            len(deck) != 52 or any(type(c) is not int for c in deck) or set(deck) != set(range(52))
        ):
            raise ValueError("Deck must be a permutation of integer card IDs 0 to 51.")
        if deck is None:
            cards = list(range(52))
            self._rng.shuffle(cards)
            deck = tuple(cards)
        self._button = next_button
        self._deck = deck
        self._cursor = 0
        self._seats = tuple((next_button + 1 + i) % 6 for i in range(6))
        self._indices = tuple(self._seats.index(i) for i in range(6))
        self._holes = [[] for _ in range(6)]
        self._board = []
        self._actions = []
        self._state = NoLimitTexasHoldem.create_state(
            _AUTOMATIONS,
            True,
            0,
            (self._small_blind, self._big_blind),
            self._big_blind,
            tuple(starting[i] for i in self._seats),
            6,
        )
        self._advance()

    def _advance(self) -> None:
        """Explicitly supply every card; engine automations settle all chip movement."""
        state = self._state
        assert state is not None
        while state.status and state.actor_index is None:
            card_id = self._deck[self._cursor]
            card = (_CARDS[card_id],)
            dealee = state.hole_dealee_index
            if dealee is not None:
                state.deal_hole(card)
                self._holes[self._seats[dealee]].append(card_id)
            elif state.can_burn_card(card):
                state.burn_card(card)
            elif state.can_deal_board(card):
                state.deal_board(card)
                self._board.append(card_id)
            else:
                raise RuntimeError("Engine stalled outside a player decision.")
            self._cursor += 1

    def observation(self) -> Observation:
        state = self._state
        seat = self.actor
        if state is None or seat is None:
            raise RuntimeError("No player decision is available.")
        assert state.street_index is not None
        holes = self._holes[seat]
        return Observation(
            seat,
            self._button,
            state.street_index,
            (holes[0], holes[1]),
            tuple(self._board),
            tuple(state.stacks[i] for i in self._indices),
            tuple(state.bets[i] for i in self._indices),
            tuple(-state.payoffs[i] for i in self._indices),
            tuple(state.statuses[i] for i in self._indices),
            tuple(state.statuses[i] and state.stacks[i] == 0 for i in self._indices),
            state.total_pot_amount,
            self._big_blind,
            LegalActions(
                state.can_fold(),
                state.can_check_or_call(),
                state.checking_or_calling_amount or 0,
                state.min_completion_betting_or_raising_to_amount,
                state.max_completion_betting_or_raising_to_amount,
            ),
            tuple(self._actions),
        )

    def is_legal(self, action: Action) -> bool:
        state = self._state
        if state is None or self.actor is None or type(action.amount) is not int:
            return False
        match action.kind:
            case "fold":
                return action.amount == 0 and state.can_fold()
            case "check_call":
                return action.amount == 0 and state.can_check_or_call()
            case "raise_to":
                return state.can_complete_bet_or_raise_to(action.amount)
            case _:
                assert_never(action.kind)

    def step(self, action: Action) -> None:
        state = self._state
        seat = self.actor
        if state is None or seat is None:
            raise RuntimeError("No player decision is available.")
        if not self.is_legal(action):
            raise ValueError("Illegal action.")
        index = self._indices[seat]
        assert state.street_index is not None
        street = state.street_index
        faced_bet = state.bets[index] < max(state.bets)
        raised: int | None = None
        match action.kind:
            case "fold":
                paid = 0
                state.fold()
            case "check_call":
                paid = state.check_or_call().amount
            case "raise_to":
                raised = action.amount
                paid = action.amount - state.bets[index]
                state.complete_bet_or_raise_to(action.amount)
            case _:
                assert_never(action.kind)
        self._actions.append(PublicAction(seat, street, action.kind, paid, raised, faced_bet))
        self._advance()

    def result(self) -> HandResult:
        state = self._state
        if state is None or state.status:
            raise RuntimeError("A completed hand is required.")
        return HandResult(
            self._button,
            tuple(self._board),
            tuple(state.payoffs[i] for i in self._indices),
            tuple(self._actions),
        )
