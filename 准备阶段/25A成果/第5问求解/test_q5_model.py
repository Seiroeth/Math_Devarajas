#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 问核心数值与约束回归测试。"""

import unittest

from q5_strict_ring import MISSILES, WARM_START, evaluate_plan, validate_plan


class Q5Benchmarks(unittest.TestCase):
    def test_missile_hit_times(self):
        self.assertAlmostEqual(MISSILES["M1"].hit_time, 66.9991708075, places=8)
        self.assertAlmostEqual(MISSILES["M2"].hit_time, 63.7503812625, places=8)
        self.assertAlmostEqual(MISSILES["M3"].hit_time, 60.3664734030, places=8)

    def test_strategy_constraints(self):
        self.assertEqual(validate_plan(WARM_START), [])

    def test_strict_three_missile_total(self):
        result = evaluate_plan(WARM_START, exact=True)
        self.assertAlmostEqual(
            result["missiles"]["M1"]["independent_duration"],
            7.6190376709,
            places=7,
        )
        self.assertAlmostEqual(
            result["missiles"]["M2"]["independent_duration"],
            10.7124720662,
            places=7,
        )
        self.assertAlmostEqual(
            result["missiles"]["M3"]["independent_duration"],
            3.7215731490,
            places=7,
        )
        self.assertAlmostEqual(result["independent_total"], 22.0530828861, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
