"""USI option precedence, odd depths and bounded actual-play searches."""
import time
import unittest

import az_usi_sampling_test as sampling


class UsiMateGuardTest(unittest.TestCase):
    setUp = sampling.UsiSamplingTest.setUp
    send = sampling.UsiSamplingTest.send
    until = sampling.UsiSamplingTest.until
    option = sampling.UsiSamplingTest.option
    search = sampling.UsiSamplingTest.search
    close = sampling.UsiSamplingTest.close

    def test_options_and_odd_rounding(self):
        self.send('usi')
        lines = self.until('usiok')
        self.assertIn('option name RootMateDepth type spin default 0 min 0 max 7', lines)
        self.assertIn('option name LeafMateDepth type spin default 3 min 0 max 7', lines)
        self.assertIn('option name LeafMateNodes type spin default 20000 min 0 max 100000000', lines)
        for value, normalized in [(0, 0), (1, 1), (2, 1), (6, 5), (7, 7), (8, 7)]:
            self.option('RootMateDepth', value)
            self.option('LeafMateDepth', value)
            _, lines = self.search(seed=42)
            settings = next(s for s in lines if s.startswith('info string go sims'))
            self.assertIn(f'root-mate-depth {normalized} ', settings)
            self.assertIn(f'leaf-mate-depth {normalized} ', settings)
            self.assertIn('probe-depth 0 ', settings)
            self.assertIn('leaf-backend ' + ('shallow' if normalized else 'off'), settings)
            self.assertEqual(any('root mate guard: depth' in s for s in lines), normalized > 0)

    def test_legacy_probe_depth_last_setting_wins(self):
        self.option('LeafMateDepth', 5)
        self.option('ProbeDepth', 0)
        _, lines = self.search(seed=42)
        self.assertTrue(any('probe-depth 0 ' in s and 'leaf-mate-depth 0 ' in s for s in lines))
        self.option('LeafMateDepth', 5)
        _, lines = self.search(seed=42)
        self.assertTrue(any('leaf-backend shallow ' in s and 'leaf-mate-depth 5 ' in s for s in lines))
        self.option('ProbeDepth', 3)
        _, lines = self.search(seed=42)
        self.assertTrue(any('probe-depth 3 ' in s and 'leaf-backend bns ' in s for s in lines))

    def test_leaf_budget_is_independent_from_legacy_bns(self):
        self.option('LeafMateDepth', 5)
        self.option('LeafMateNodes', 1234)
        self.option('ProbeNodes', 17)
        _, lines = self.search(seed=42)
        self.assertTrue(any('leaf-mate-nodes 1234 ' in s and 'probe-nodes 17 ' in s for s in lines))

    def test_movetime_and_stop_remain_responsive(self):
        self.option('RootMateDepth', 7)
        self.option('RootMateNodes', 100000000)
        self.option('RootMateTimeMs', 60000)
        self.send('position sfen ' + sampling.INITIAL)
        start = time.monotonic()
        self.send('go movetime 80')
        lines = self.until('bestmove ')
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(sum(s.startswith('bestmove ') for s in lines), 1)
        self.send('go infinite')
        time.sleep(0.05)
        start = time.monotonic()
        self.send('stop')
        self.until('bestmove ')
        self.assertLess(time.monotonic() - start, 2)
        # A subsequent command sees a joined search and a fresh stop flag.
        self.search(seed=42)

    def test_zero_root_nodes_disables_guard(self):
        self.option('RootMateDepth', 7)
        self.option('RootMateNodes', 0)
        _, lines = self.search(seed=42)
        self.assertFalse(any('root mate guard:' in s for s in lines))


if __name__ == '__main__':
    unittest.main()
