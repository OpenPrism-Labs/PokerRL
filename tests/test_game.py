"""Independent contracts for Kuhn legality, observation privacy, and chip accounting."""

import unittest

from pokerrl.game import DEALS, TERMINALS, InvalidState, State


class GameTests(unittest.TestCase):
    def test_complete_tree_has_legal_actions_and_zero_sum_payouts(self):
        expected = {
            "": ("check", "bet"),
            "k": ("check", "bet"),
            "b": ("fold", "call"),
            "kb": ("fold", "call"),
        }
        for cards in DEALS:
            with self.subTest(cards=cards):
                leaves = {}

                def traverse(state, leaves):
                    if state.terminal:
                        leaves[state.history] = state.rewards()
                        with self.assertRaises(InvalidState):
                            state.step("check")
                        return
                    self.assertEqual(state.legal_actions, expected[state.history])
                    self.assertEqual(state.player, len(state.history) % 2)
                    for action in state.legal_actions:
                        traverse(state.step(action), leaves)
                    illegal = "call" if "check" in state.legal_actions else "check"
                    with self.assertRaises(InvalidState):
                        state.step(illegal)
                    with self.assertRaises(InvalidState):
                        state.rewards()

                traverse(State(cards), leaves)
                sign = 1 if cards[0] > cards[1] else -1
                self.assertEqual(set(leaves), set(TERMINALS))
                self.assertEqual(
                    leaves,
                    {
                        "kk": (sign, -sign),
                        "bf": (1, -1),
                        "bc": (2 * sign, -2 * sign),
                        "kbf": (-1, 1),
                        "kbc": (2 * sign, -2 * sign),
                    },
                )

    def test_observations_hide_opponent_private_card(self):
        for history, first, second in (
            ("", (0, 1), (0, 2)),
            ("k", (0, 1), (2, 1)),
            ("kb", (0, 1), (0, 2)),
        ):
            with self.subTest(history=history):
                self.assertEqual(
                    State(first, history).observation(), State(second, history).observation()
                )
        self.assertEqual(State((0, 1)).observation().card, 0)

    def test_boundary_rejects_invalid_deals_and_histories(self):
        for cards, history in (
            ((0, 0), ""),
            ((-1, 2), ""),
            ((0, 3), ""),
            ((0, 1), "bb"),
            ((0, 1), "f"),
        ):
            with self.subTest(cards=cards, history=history), self.assertRaises(InvalidState):
                State(cards, history)
