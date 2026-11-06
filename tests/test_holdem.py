"""Complete hands with independently computed payouts and public-only decisions."""

import random
import unittest
from dataclasses import FrozenInstanceError

from pokerrl.holdem import Action, HoldemEnv


def deck_for(holes: tuple[str, ...], board: str) -> tuple[int, ...]:
    """Arrange test cards in physical dealing order, with distinct unused burns."""

    def ids(text: str) -> list[int]:
        return [
            4 * "23456789TJQKA".index(text[i]) + "cdhs".index(text[i + 1])
            for i in range(0, len(text), 2)
        ]

    hands = [ids(hand) for hand in holes]
    community = ids(board)
    reserved = [c for hand in hands for c in hand] + community
    assert len(set(reserved)) == len(reserved)
    unused = [i for i in range(52) if i not in reserved]
    prefix = [hand[j] for j in range(2) for hand in hands]
    prefix += [unused[0], *community[:3], unused[1], community[3], unused[2], community[4]]
    return tuple(prefix + unused[3:])


def finish_calls(env: HoldemEnv) -> None:
    for _ in range(100):
        if env.terminal:
            return
        env.step(Action("check_call"))
    raise AssertionError("Hand failed to terminate within 100 calls.")


class HoldemTests(unittest.TestCase):
    def test_round_robin_dealing_actor_order_and_public_contributions(self):
        env = HoldemEnv(7)
        env.reset(deck=tuple(range(52)))
        for seat in (2, 3, 4, 5, 0, 1):
            obs = env.observation()
            self.assertEqual((obs.seat, obs.hole, obs.street), (seat, (seat, seat + 6), 0))
            env.step(Action("check_call"))
        obs = env.observation()
        self.assertEqual((obs.seat, obs.board), (0, (13, 14, 15)))
        self.assertEqual((obs.pot, obs.contributions, obs.bets), (12, (2,) * 6, (0,) * 6))
        self.assertFalse(obs.legal.can_fold)
        finish_calls(env)
        self.assertEqual(env.result().board, (13, 14, 15, 17, 19))
        self.assertEqual([a.seat for a in env.result().actions[6:]], list(range(6)) * 3)
        self.assertEqual(sum(env.result().payoffs), 0)

    def test_folds_settle_blinds_and_rotation_preserves_external_seats(self):
        env = HoldemEnv(8, button=3)
        for button in (3, 4, 5, 0, 1, 2, 3):
            env.reset()
            self.assertEqual(env.observation().button, button)
            self.assertEqual(env.actor, (button + 3) % 6)
            for _ in range(5):
                env.step(Action("fold"))
            expected = [0] * 6
            expected[(button + 1) % 6] = -1
            expected[(button + 2) % 6] = 1
            self.assertEqual(env.result().payoffs, tuple(expected))
            self.assertEqual(env.result().board, ())
            self.assertIsNone(env.actor)

    def test_privacy_hides_changed_opponent_cards_and_snapshots_are_immutable(self):
        first, second = HoldemEnv(0), HoldemEnv(1)
        deck = list(range(52))
        deck[0], deck[40] = deck[40], deck[0]
        first.reset(deck=tuple(range(52)))
        second.reset(deck=tuple(deck))
        obs = first.observation()
        self.assertEqual(obs, second.observation())
        with self.assertRaises(FrozenInstanceError):
            obs.seat = 4
        first.step(Action("check_call"))
        self.assertEqual((obs.actions, obs.pot, obs.hole), ((), 3, (2, 8)))
        finish_calls(first)
        finish_calls(second)
        self.assertFalse(hasattr(first.result(), "hole"))

    def test_seeded_trajectories_ignore_global_rng_and_deck_override(self):
        self.addCleanup(random.setstate, random.getstate())
        first, second = HoldemEnv(45), HoldemEnv(45)
        for _ in range(3):
            first.reset()
            random.seed(999)
            second.reset()
            while not first.terminal:
                self.assertEqual(first.observation(), second.observation())
                first.step(Action("check_call"))
                second.step(Action("check_call"))
            self.assertEqual(first.result(), second.result())
        third = HoldemEnv(45)
        third.reset(deck=tuple(range(52)))
        finish_calls(third)
        third.reset(button=5)
        fresh = HoldemEnv(45)
        fresh.reset()
        self.assertEqual(third.observation(), fresh.observation())
        finish_calls(third)
        finish_calls(fresh)

    def test_sidepots_and_uncalled_excess_have_independent_winners(self):
        # AA wins 3*20, KK wins 2*30, QQ gets its unmatched 50 back.
        deck = deck_for(
            ("AcAd", "KcKd", "QcQd", "JcJd", "TcTd", "9c9d"),
            "2c3d7h8s4c",
        )
        env = HoldemEnv(0, stacks=(20, 50, 100, 200, 200, 200))
        env.reset(deck=deck)
        env.step(Action("raise_to", 100))
        for _ in range(3):
            env.step(Action("fold"))
        env.step(Action("check_call"))
        self.assertEqual(env.observation().all_in, (True, False, True, False, False, False))
        env.step(Action("check_call"))
        self.assertTrue(env.terminal)
        self.assertEqual(env.result().payoffs, (40, 10, -50, 0, 0, 0))
        self.assertEqual(len(env.result().board), 5)
        self.assertEqual([a.paid for a in env.result().actions], [100, 0, 0, 0, 19, 48])

    def test_short_allin_does_not_reopen_prior_raiser(self):
        env = HoldemEnv(1, stacks=(200, 200, 200, 15, 200, 200))
        env.reset(
            deck=deck_for(
                ("9c9d", "TcTd", "AcAd", "KcKd", "QcQd", "JcJd"),
                "2c3d7h8s4c",
            )
        )
        env.step(Action("raise_to", 10))
        self.assertEqual(env.observation().legal.min_raise_to, 15)
        env.step(Action("raise_to", 15))
        self.assertEqual(env.observation().legal.min_raise_to, 23)
        env.step(Action("check_call"))
        for _ in range(3):
            env.step(Action("fold"))
        self.assertEqual(env.actor, 2)
        self.assertIsNone(env.observation().legal.min_raise_to)
        self.assertFalse(env.is_legal(Action("raise_to", 30)))
        finish_calls(env)
        self.assertEqual(env.result().payoffs, (-1, -2, 33, -15, -15, 0))

    def test_cumulative_short_allins_reopen_a_full_raise(self):
        env = HoldemEnv(1, stacks=(200, 200, 200, 15, 20, 200))
        env.reset()
        for amount in (10, 15, 20):
            env.step(Action("raise_to", amount))
        env.step(Action("check_call"))
        env.step(Action("fold"))
        env.step(Action("fold"))
        self.assertEqual(env.actor, 2)
        self.assertEqual(env.observation().legal.min_raise_to, 28)
        self.assertTrue(env.is_legal(Action("raise_to", 31)))
        env.step(Action("raise_to", 31))
        finish_calls(env)
        self.assertEqual(sum(env.result().payoffs), 0)

    def test_board_tie_awards_odd_chip_clockwise_from_button(self):
        env = HoldemEnv(2)
        env.reset(
            deck=deck_for(
                ("2c2d", "3c3d", "4c4d", "5c5d", "6c6d", "7c7d"),
                "TsJsQsKsAs",
            )
        )
        env.step(Action("raise_to", 5))
        env.step(Action("check_call"))
        for _ in range(4):
            env.step(Action("fold"))
        finish_calls(env)
        # Thirteen chips: tied seats 2 and 3 get seven and six, respectively.
        self.assertEqual(env.result().payoffs, (-1, -2, 2, 1, 0, 0))

    def test_arbitrary_integer_raise_and_illegal_action_leave_hand_unchanged(self):
        env = HoldemEnv(0)
        env.reset()
        before = env.observation()
        for action in (Action("raise_to", 3), Action("raise_to", 201), Action("fold", 1)):
            self.assertFalse(env.is_legal(action))
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.observation(), before)
        env.step(Action("raise_to", 17))
        self.assertEqual(env.observation().legal.min_raise_to, 32)
        finish_calls(env)
        self.assertEqual(sum(env.result().payoffs), 0)

    def test_lifecycle_and_reset_overrides(self):
        env = HoldemEnv(0)
        with self.assertRaises(RuntimeError):
            env.result()
        with self.assertRaises(RuntimeError):
            env.observation()
        with self.assertRaises(RuntimeError):
            env.step(Action("check_call"))
        env.reset(button=2, stacks=(50,) * 6)
        self.assertEqual(env.observation().stacks, (50, 50, 50, 49, 48, 50))
        with self.assertRaises(RuntimeError):
            env.reset()
        with self.assertRaises(RuntimeError):
            env.result()
        finish_calls(env)
        with self.assertRaises(RuntimeError):
            env.step(Action("check_call"))
        for invalid in (tuple(range(51)), (0,) * 52, tuple(range(1, 53))):
            with self.assertRaises(ValueError):
                env.reset(deck=invalid)
        env.reset()
        self.assertEqual(env.observation().button, 3)
        self.assertEqual(env.observation().stacks, (200, 200, 200, 200, 199, 198))
        finish_calls(env)

    def test_boundary_rejects_bad_stack_button_blind_and_card_types(self):
        for stacks in ((100,) * 5, (0,) * 6, (-1,) * 6, (True,) * 6):
            with self.assertRaises(ValueError):
                HoldemEnv(0, stacks=stacks)
        for button in (-1, 6, True):
            with self.assertRaises(ValueError):
                HoldemEnv(0, button=button)
        for small, big in ((0, 2), (3, 2), (1, True)):
            with self.assertRaises(ValueError):
                HoldemEnv(0, small_blind=small, big_blind=big)
        env = HoldemEnv(0)
        with self.assertRaises(ValueError):
            env.reset(deck=(False, *range(1, 52)))
        env.reset(stacks=(1,) * 6)
        finish_calls(env)
        self.assertEqual(sum(env.result().payoffs), 0)
