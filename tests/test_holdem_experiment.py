"""Real-session evaluation contracts, independent of neural training quality."""

import unittest

from pokerrl.holdem_experiment import evaluate, summarize
from pokerrl.holdem_learning import ScriptedPolicy


class HoldemExperimentTests(unittest.TestCase):
    def test_each_hero_plays_every_button_position_in_complete_sessions(self):
        rows, traces = evaluate(ScriptedPolicy("tight", 0.3), "tight", 17, 1, 36, 18)
        for scenario in ("fixed", "shifting"):
            selected = [row for row in rows if row.scenario == scenario]
            self.assertEqual(len(selected), 36)
            for hero in range(6):
                positions = {(row.hero - row.button) % 6 for row in selected if row.hero == hero}
                self.assertEqual(positions, set(range(6)))
        for trace in traces:
            self.assertEqual(sum(trace.payoffs), 0)

    def test_identical_policies_have_zero_paired_session_difference(self):
        history, _ = evaluate(ScriptedPolicy("loose", 0.3), "history", 5, 2, 4, 2)
        zero, _ = evaluate(ScriptedPolicy("loose", 0.3), "zero_history", 5, 2, 4, 2)
        fixed, _ = evaluate(ScriptedPolicy("tight", 0.3), "tight", 5, 2, 4, 2)
        for entry in summarize(history + zero + fixed):
            if entry["algorithm"] == "history_minus_zero":
                self.assertEqual(entry["bb_per_hand"], 0)
                self.assertEqual(entry["ci95_low"], 0)
                self.assertEqual(entry["ci95_high"], 0)
