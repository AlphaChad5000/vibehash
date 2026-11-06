import unittest

from vibehash import cli


class FakeNode:
    def __init__(self, genesis):
        self.genesis = genesis

    def get(self, path):
        assert path == "/genesis"
        if isinstance(self.genesis, Exception):
            raise self.genesis
        return self.genesis


class EvoNetworkViewTests(unittest.TestCase):
    """The miner's screen opens on the Evo scene on a network whose genesis takes Evo jobs (cli.evo_network)."""

    def test_an_evo_network_opens_on_the_evo_scene(self):
        self.assertTrue(cli.evo_network(FakeNode({"chain_id": "vibehash-evo-1", "es": True})))

    def test_other_networks_and_trouble_open_on_the_usual_one(self):
        for g in ({"chain_id": "vibehash-public-1"}, {"es": False}, {"es": "yes"}, OSError("unreachable")):
            self.assertFalse(cli.evo_network(FakeNode(g)), g)


if __name__ == "__main__":
    unittest.main()
