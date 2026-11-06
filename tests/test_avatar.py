"""The floating avatar (avatar.py): what it shows for what the feed says, its menu, and its drawing, with no screen:
tkinter is played by a fake, and only `run` (which needs a display) is left out."""
import os
import tempfile
import unittest
from unittest import mock

from vibehash import app, avatar

GOLD = avatar.GOLD


def snapshot(activity="training unit 1 of 3", busy=True, phase="compute", earned=1.5):
    return {"workers": [{"activity": activity, "busy": busy, "phase": phase, "session": {"earned_vibe": earned}}]}


class MoodTests(unittest.TestCase):
    def test_each_phase_has_a_look(self):
        m, earned, paid = avatar.mood_of(None, None, 100.0, None)
        self.assertEqual((m.name, m.speed), ("offline", 0.0))
        m, earned, paid = avatar.mood_of(snapshot(), None, 100.0, None)
        self.assertEqual((m.name, m.colour, earned), ("training", GOLD, 1.5))
        self.assertGreater(m.speed, 1)
        m, *_ = avatar.mood_of(snapshot("checking other workers' results (1 of 2)"), 1.5, 101.0, None)
        self.assertEqual(m.name, "checking")
        m, *_ = avatar.mood_of(snapshot("uploading unit 2 of 3"), 1.5, 101.0, None)
        self.assertEqual(m.name, "uploading")
        m, *_ = avatar.mood_of(snapshot("registered; waiting for work", busy=False, phase="register"), 1.5, 101.0, None)
        self.assertEqual((m.name, m.label), ("resting", "registered; waiting for work"))
        self.assertLess(m.speed, 1)

    def test_a_payment_shows_as_sparks_for_a_while_then_passes(self):
        m, earned, paid = avatar.mood_of(snapshot(earned=1.5), 1.5, 100.0, None)
        self.assertEqual(m.name, "training")
        m, earned, paid = avatar.mood_of(snapshot(earned=1.62), 1.5, 101.0, None)  # the round paid
        self.assertEqual((m.name, m.label, paid), ("paid", "paid +0.1200 VIBE", 101.0))
        m, earned, paid = avatar.mood_of(snapshot(earned=1.62), 1.62, 101.0 + avatar.PAYDAY_S - 0.5, paid)
        self.assertEqual(m.name, "paid")  # still celebrating
        m, earned, paid = avatar.mood_of(snapshot(earned=1.62), 1.62, 101.0 + avatar.PAYDAY_S + 0.5, paid)
        self.assertEqual(m.name, "training")  # and back to work
        m, earned, paid = avatar.mood_of(snapshot(earned=None), 1.62, 200.0, paid)  # a feed without the round log
        self.assertEqual((m.name, earned), ("training", 1.62))


class FakeTk:
    """Enough of tkinter for the avatar to be built and drawn."""

    class Canvas:
        def __init__(self, root, **kw):
            self.items, self.bindings, self.kw = [], {}, kw

        def pack(self): pass
        def bind(self, event, fn): self.bindings[event] = fn
        def delete(self, what): self.items.clear()
        def create_polygon(self, pts, **kw): self.items.append(("polygon", pts, kw))
        def create_line(self, *a, **kw): self.items.append(("line", a, kw))
        def create_oval(self, *a, **kw): self.items.append(("oval", a, kw))
        def create_text(self, *a, **kw): self.items.append(("text", a, kw))

    class Menu:
        def __init__(self, root, tearoff=0):
            self.entries = []

        def add_command(self, label, command=None, state=None): self.entries.append((label, command, state))
        def add_separator(self): self.entries.append(None)
        def tk_popup(self, x, y): self.popped = (x, y)
        def grab_release(self): pass


class FakeRoot:
    def __init__(self):
        self.attrs, self.geometry_calls, self.destroyed = {}, [], False

    def overrideredirect(self, v): self.override = v
    def attributes(self, name, value): self.attrs[name] = value
    def config(self, **kw): pass
    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080
    def winfo_x(self): return 100
    def winfo_y(self): return 200
    def geometry(self, g): self.geometry_calls.append(g)
    def destroy(self): self.destroyed = True


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.root = FakeRoot()
        self.av = avatar.Avatar("http://127.0.0.1:7545/", FakeTk, self.root, now=lambda: 100.0)

    def test_it_is_a_small_topmost_frameless_window_in_the_corner(self):
        self.assertTrue(self.root.override)
        self.assertTrue(self.root.attrs.get("-topmost"))
        self.assertRegex(self.root.geometry_calls[0], rf"^{avatar.SIZE}x{avatar.SIZE}\+\d+\+\d+$")
        self.assertEqual(self.av.feed, "http://127.0.0.1:7545")

    def test_the_mark_is_three_blocks_in_the_moods_colour(self):
        polygons = [i for i in self.av.canvas.items if i[0] == "polygon"]
        self.assertEqual(len(polygons), 3)
        self.assertEqual({p[2]["fill"] for p in polygons}, {avatar.GREY})  # offline until the feed answers
        with mock.patch.object(avatar, "fetch", return_value=snapshot()):
            self.av.tick()
        polygons = [i for i in self.av.canvas.items if i[0] == "polygon"]
        self.assertEqual({p[2]["fill"] for p in polygons}, {GOLD})
        self.assertEqual(self.av.mood.name, "training")

    def test_a_drag_moves_it_and_is_remembered_a_click_opens_the_window(self):
        press = mock.Mock(x=10, y=10)
        self.av._press(press)
        self.av._move(mock.Mock(x_root=500, y_root=400))
        self.assertEqual(self.root.geometry_calls[-1], "+490+390")
        self.av._release(mock.Mock(x=40, y=40))
        self.assertEqual(avatar.saved_place(), (100, 200))  # where the window says it is
        with mock.patch.object(avatar.webbrowser, "open") as opened:
            self.av._press(mock.Mock(x=10, y=10))
            self.av._release(mock.Mock(x=11, y=10))  # barely moved: a click
        opened.assert_called_once_with("http://127.0.0.1:7545/dashboard")

    def test_the_menu_fits_the_app(self):
        labels = [i[0] for i in self.av.menu_items() if i]
        self.assertIn("Open the window", labels)
        self.assertNotIn("Pause mining", labels)  # no app running
        self.assertNotIn("Stop mining", labels)
        app.app_dir().mkdir(parents=True)
        app.state_path().write_text('{"beat": 1e12, "started": 1, "paused": false}')
        labels = [i[0] for i in self.av.menu_items() if i]
        self.assertIn("Pause mining", labels)
        self.assertIn("Stop mining", labels)
        dict((i for i in self.av.menu_items() if i))["Pause mining"]()
        self.assertEqual(app.control_path().read_text().strip(), "pause")
        app.state_path().write_text('{"beat": 1e12, "started": 1, "paused": true}')
        self.assertIn("Resume mining", [i[0] for i in self.av.menu_items() if i])
        self.av._menu(mock.Mock(x_root=5, y_root=5))  # builds the tk menu without complaint

    def test_hide_closes_the_window(self):
        self.av.hide()
        self.assertTrue(self.root.destroyed)
        self.assertTrue(self.av.hidden)


if __name__ == "__main__":
    unittest.main()
