#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from q4_strict_ring import REFERENCE, REFINED, decode, independent_union


class Q4Benchmarks(unittest.TestCase):
    def test_reference(self):
        self.assertAlmostEqual(independent_union(decode(REFERENCE), exact=True)[2], 10.6125081507, places=7)

    def test_refined(self):
        smokes = decode(REFINED)
        self.assertTrue(all(smoke.valid for smoke in smokes))
        self.assertAlmostEqual(independent_union(smokes, exact=True)[2], 11.1417943151, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
