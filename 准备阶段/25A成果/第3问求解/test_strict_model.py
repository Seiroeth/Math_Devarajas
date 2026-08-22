#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公共严格模型的三个回归基准。"""

import math
import unittest

from q3_strict_ring import REFINED, decode
from smoke_strict_core import Smoke, exact_intervals, independent_union, interval_length


class StrictModelBenchmarks(unittest.TestCase):
    def test_q1(self):
        smoke = Smoke(math.pi, 120.0, 1.5, 3.6)
        self.assertAlmostEqual(interval_length(exact_intervals(smoke)), 1.3916426683, places=7)

    def test_q2(self):
        smoke = Smoke(math.radians(176.643), 70.0, 0.0, 2.49705)
        self.assertAlmostEqual(interval_length(exact_intervals(smoke)), 4.5428788269, places=7)

    def test_q3_refined(self):
        smokes = decode(REFINED)
        for left, right in zip(smokes[:-1], smokes[1:]):
            self.assertGreaterEqual(right.td - left.td, 1.0)
        self.assertAlmostEqual(independent_union(smokes, exact=True)[2], 7.6190376709, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
