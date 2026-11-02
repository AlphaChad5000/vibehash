import unittest
from types import SimpleNamespace as U

from vibehash.coordinator import _affinity, _piece, _spread


class PrepAssignTests(unittest.TestCase):
    def test_a_round_takes_units_across_pieces(self):
        ready = [U(name=f"layers-00-03-c{c:05d}") for c in range(10)] + [U(name=f"layers-04-07-c{c:05d}") for c in range(3)]
        names = [u.name for u in _spread(ready)][:6]
        self.assertEqual([_piece(n) for n in names], ["layers-00-03", "layers-04-07"] * 3)
        self.assertEqual(names[0], "layers-00-03-c00000")  # each piece keeps its own order

    def test_a_piece_stays_with_the_same_miners(self):
        miners = [f"vibe1{i:039d}" for i in range(6)]
        first = min(miners, key=lambda a: _affinity("layers-08-11", a))
        self.assertEqual(first, min(reversed(miners), key=lambda a: _affinity("layers-08-11", a)))  # not order-bound
        spread = {min(miners, key=lambda a: _affinity(f"layers-{4 * i:02d}-{4 * i + 3:02d}", a)) for i in range(9)}
        self.assertGreater(len(spread), 1)  # pieces don't all land on one miner
        self.assertEqual(_piece("rows-01-007"), "rows-01-007")


if __name__ == "__main__":
    unittest.main()
