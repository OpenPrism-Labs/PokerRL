"""Matched session aggregation with independently specified returns."""

import unittest

from pokerrl.algorithm_experiment import summarize
from pokerrl.holdem_experiment import EvaluationRow


class AlgorithmExperimentTests(unittest.TestCase):
    def test_paired_session_means_use_matching_opponent_runs(self):
        rows = []
        for scenario in ("fixed", "shifting"):
            for session, rewards in enumerate(((2.0, 6.0), (10.0, 14.0))):
                for hand, phase in enumerate(("before", "after")):
                    for algorithm, payoff in (
                        ("ppo", rewards[hand]),
                        ("reinforce", 2),
                        ("tight", 0),
                    ):
                        rows.append(
                            EvaluationRow(
                                algorithm,
                                0,
                                session,
                                scenario,
                                hand,
                                0,
                                5,
                                phase,
                                payoff,
                                1,
                            )
                        )
        output = summarize(rows, ("ppo", "reinforce"))
        paired = next(
            row
            for row in output
            if row["scenario"] == "fixed"
            and row["phase"] == "all"
            and row["algorithm"] == "ppo_minus_reinforce"
        )
        self.assertEqual(paired["bb_per_hand"], 6)
        self.assertEqual(paired["ci95_low"], 2)
        self.assertEqual(paired["ci95_high"], 10)
        self.assertEqual(paired["sessions"], 2)
