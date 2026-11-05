"""The hardware census: a worker registers what kind of machine it is, in a canonical form the node checks and the
record keeps, and /network counts the classes without naming anyone."""
import unittest

from vibehash import compute, hardware, netstats
from vibehash.coordinator import Phase

from tests.test_record_v2 import network


class HardwareClassTests(unittest.TestCase):
    def test_this_computer_has_a_valid_class(self):
        klass = hardware.hardware_class(compute.NumpyBackend())
        self.assertTrue(hardware.valid_class(klass), klass)
        self.assertEqual(klass["vendor"], "cpu")
        self.assertIn("threads", klass["label"])
        self.assertIn("GB", hardware.class_name(klass))

    def test_card_names_are_shortened_the_way_people_say_them(self):
        self.assertEqual(hardware.short_label("NVIDIA GeForce RTX 4060 Laptop GPU"), "RTX 4060 Laptop")
        self.assertEqual(hardware.short_label("AMD Radeon RX 7900 XTX"), "RX 7900 XTX")
        self.assertEqual(hardware.short_label("Intel(R) Arc(TM) A770 Graphics"), "Arc A770")
        self.assertEqual(hardware.short_label(""), "GPU")
        self.assertLessEqual(len(hardware.short_label("x" * 200)), hardware.MAX_LABEL)

    def test_a_rocm_arch_loses_its_feature_flags(self):
        for flags in ("gfx1100:sramecc+:xnack-", "gfx1100:xnack+", "gfx1100"):
            self.assertEqual(hardware._canonical("amd", flags, 24, "AMD Radeon RX 7900 XTX"),
                             {"vendor": "amd", "arch": "gfx1100", "memory_gb": 24, "label": "RX 7900 XTX"})

    def test_only_the_canonical_form_is_valid(self):
        good = {"vendor": "nvidia", "arch": "sm_89", "memory_gb": 8, "label": "RTX 4060"}
        self.assertTrue(hardware.valid_class(good))
        for bad in (
            {**good, "vendor": "nvidia corp"},  # not a vendor the code knows
            {**good, "label": "RTX 9090 Ti <script>"},  # characters outside the whitelist
            {**good, "label": " RTX 4060"},  # not stripped: two spellings of one card
            {**good, "label": "x" * 41},
            {**good, "arch": "SM_89"},  # upper case: again two spellings
            {**good, "memory_gb": 8.0},  # a float
            {**good, "memory_gb": True},
            {**good, "memory_gb": 5000},
            {**good, "serial": "1234"},  # anything extra
            {k: v for k, v in good.items() if k != "arch"},
            "RTX 4060", None, [],
        ):
            self.assertFalse(hardware.valid_class(bad), bad)

    def test_a_torch_device_gets_a_class(self):
        if not compute.torch_available():
            self.skipTest("no PyTorch")
        for d in compute.list_devices():
            if d.id == "cpu":
                continue
            klass = hardware.hardware_class(compute.TorchBackend(d.id))
            self.assertTrue(hardware.valid_class(klass), klass)
            self.assertNotEqual(klass["vendor"], "cpu")


class RegistrationTests(unittest.TestCase):
    def register(self, proto, w, hardware_class):
        c = proto.coord
        r = c.r
        body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb}
        if hardware_class is not None:
            body["hardware"] = hardware_class
        return c.register(w.address, w.hardware.flops_per_sec, w.hardware.memory_mb, envelope=proto._signed(w, "register", body))

    def test_the_class_is_kept_from_the_signed_body_and_junk_is_refused(self):
        proto, _ = network(3)
        c = proto.coord
        c.open_round()
        w1, w2, w3 = proto.workers.values()
        klass = {"vendor": "nvidia", "arch": "sm_89", "memory_gb": 8, "label": "RTX 4060"}
        entry = self.register(proto, w1, klass)
        self.assertEqual(c.r.regs[w1.address].hardware, klass)
        self.assertEqual(entry["body"]["hardware"], klass)  # in the record, as signed
        self.register(proto, w2, None)  # an older worker says nothing: still registered
        self.assertIsNone(c.r.regs[w2.address].hardware)
        # A made-up card: the node refuses it at the door (tests/test_dos.py), and the coordinator, which replays
        # records sealed by nodes of every version, counts it as no class rather than refusing the round
        self.register(proto, w3, {"vendor": "nvidia", "arch": "sm_89", "memory_gb": 8, "label": "RTX 9090 Ti!!"})
        self.assertIsNone(c.r.regs[w3.address].hardware)
        self.assertEqual(c.phase, Phase.REGISTER)

    def test_a_round_with_classes_replays(self):
        """The class changes the record (the signed body carries it) but not what the round does, and a replay sees it."""
        import copy

        from vibehash.replay import replay_block

        proto, _ = network(3)
        klass = {"vendor": "cpu", "arch": "amd64", "memory_gb": 32, "label": "CPU 24 threads"}
        signed = proto._signed

        def with_class(w, kind, body):
            return signed(w, kind, {**body, "hardware": klass} if kind == "register" else body)
        proto._signed = with_class
        before = copy.deepcopy(proto.chain)
        rep = proto.run_round()
        self.assertEqual(len(rep.record["registrations"]), 3)
        self.assertTrue(all(e["body"]["hardware"] == klass for e in rep.record["registrations"]))
        self.assertEqual(replay_block(copy.deepcopy(before), rep.block, rep.record), [])


class CensusTests(unittest.TestCase):
    def test_counts_classes_of_todays_workers_and_who_is_online(self):
        now = 100_000.0
        rtx = {"vendor": "nvidia", "arch": "sm_89", "memory_gb": 8, "label": "RTX 4060"}
        m3 = {"vendor": "apple", "arch": "mps", "memory_gb": 16, "label": "Apple M3"}
        cpu = {"vendor": "cpu", "arch": "amd64", "memory_gb": 32, "label": "CPU 24 threads"}
        rounds = [
            {"timestamp": now - 2 * netstats.DAY_S, "workers": {"old": rtx}},  # more than a day ago: not counted
            {"timestamp": now - 3600, "workers": {"a": rtx, "b": m3, "c": None}},  # c runs an older worker
            {"timestamp": now - 300, "workers": {"a": rtx, "d": rtx, "e": cpu}},
            {"timestamp": now - 30, "workers": {"b": cpu}},  # b moved to another machine: counted where it is now
        ]
        c = netstats.census(rounds, now)
        self.assertEqual((c["online"], c["day"], c["reported"]), (4, 5, 4))
        self.assertEqual([(k["label"], k["online"], k["day"]) for k in c["classes"]],
                         [("CPU 24 threads", 2, 2), ("RTX 4060", 2, 2)])  # b left the M3, so it isn't listed; ties by name
        self.assertEqual(c["vendors"], {"nvidia": 2, "cpu": 2})
        self.assertEqual(c["classes"][1]["arch"], "sm_89")
        for klass in c["classes"]:  # nothing that names a worker
            self.assertEqual(set(klass), {"label", "vendor", "arch", "memory_gb", "online", "day"})

    def test_nothing_registered(self):
        self.assertEqual(netstats.census([], 1.0), {"online": 0, "day": 0, "reported": 0, "classes": [], "vendors": {}})


if __name__ == "__main__":
    unittest.main()
