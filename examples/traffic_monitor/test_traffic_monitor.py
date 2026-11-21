"""Offline tests for traffic_monitor.py.

Mirage is replaced by a fake that "sees" a white rectangle in a synthetic
video, so these run without the engine, the weights or a camera:

    python -m pytest examples/traffic_monitor
"""
import csv
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import traffic_monitor as tm  # noqa: E402


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.setattr(tm, "APPROACH", "down")
    monkeypatch.setattr(tm, "BBOX_ORDER", "xyxy")
    monkeypatch.setattr(tm, "SHOW_WINDOW", False)
    monkeypatch.setattr(tm, "PLATE_REGEX", tm.ANY_PLATE_REGEX)   # tests below use "34 ABC 123"
    monkeypatch.setattr(tm, "READ_PLATES", True)                  # most tests below check plates too


def test_boxes_scale_order_and_clamp(monkeypatch):
    assert tm.to_pixels([100, 200, 500, 1000], 1000, 500) == (100, 100, 500, 499)
    assert tm.to_pixels([0.1, 0.2, 0.5, 0.6], 1000, 1000) == (100, 200, 500, 600)
    monkeypatch.setattr(tm, "BBOX_ORDER", "yxyx")
    assert tm.to_pixels([200, 100, 600, 500], 1000, 1000) == (100, 200, 500, 600)


def test_parser_accepts_several_shapes():
    result = {
        "objects": [
            {"label": "Car", "bbox": [100, 100, 300, 300], "score": 0.9},
            {"name": "tree", "boxes": [[0, 0, 50, 50], [60, 60, 120, 120]]},
            {"class": "truck", "box": {"x0": 400, "y0": 400, "x1": 600, "y1": 600}},
        ],
        "text": [{"text": "34 ABC 123", "bbox": [150, 250, 250, 280]}, "loose line"],
    }
    objects = tm.parse_objects(result, 1000, 1000)
    assert [o[0] for o in objects] == ["car", "tree", "tree", "truck"]
    assert [o[0] for o in objects if tm.is_vehicle(o[0])] == ["car", "truck"]
    assert tm.parse_text(result, 1000, 1000) == [("34 ABC 123", (150, 250, 250, 280)), ("loose line", None)]
    assert tm.parse_objects({}, 10, 10) == [] and tm.parse_text({"text": None}, 10, 10) == []


def test_plate_cleaning_and_ocr_fixes(monkeypatch):
    assert tm.plate_candidates("34 abc-123") == ["34ABC123"]
    assert tm.plate_candidates("STOP") == []            # no digit, never "5TOP"
    assert tm.plate_candidates("SPEED LIMIT 50") == []  # too long
    monkeypatch.setattr(tm, "PLATE_REGEX", r"^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$")
    assert tm.plate_candidates("34 ABC I23") == ["34ABC123"]  # I read as 1


@pytest.mark.parametrize("approach,dy,expected", [("down", +0.3, "coming"), ("down", -0.3, "going"),
                                                  ("up", +0.3, "going"), ("down", 0.0, "unknown")])
def test_direction(monkeypatch, approach, dy, expected):
    monkeypatch.setattr(tm, "APPROACH", approach)
    track = tm.Track(1, "car", (0, 0, 1, 1), 0, 0, history=[(0.5, 0.4, 0.05), (0.5, 0.4 + dy, 0.05)])
    assert track.direction() == expected


def test_tracker_keeps_identity_and_finishes_lost_tracks(monkeypatch):
    monkeypatch.setattr(tm, "MAX_MISSED", 1)
    tracker = tm.Tracker()
    a, _ = tracker.update([("car", None, (100, 100, 200, 200))], 1000, 1000, 0)
    b, _ = tracker.update([("car", None, (110, 110, 210, 210))], 1000, 1000, 1)
    assert a[0].id == b[0].id and len(b[0].history) == 2
    tracker.update([], 1000, 1000, 2)
    _, finished = tracker.update([], 1000, 1000, 3)
    assert [t.id for t in finished] == [a[0].id] and not tracker.tracks


def test_rtsp_password_is_hidden():
    assert tm.redact("rtsp://admin:s3cr@t@10.0.0.2:554/x") == "rtsp://admin:***@10.0.0.2:554/x"


class FakeMirage:
    """Finds the white rectangle and 'reads' a plate inside it."""

    def __init__(self, weights=None):
        pass

    def describe(self, image, *, sections, **_):
        mask = image[:, :, 0] > 200
        ys, xs = np.nonzero(mask)
        h, w = image.shape[:2]
        out = {"objects": [], "text": []}
        if len(xs) == 0:
            return out
        box = [xs.min() / w * 1000, ys.min() / h * 1000, xs.max() / w * 1000, ys.max() / h * 1000]
        if "objects" in sections:
            out["objects"].append({"id": "car", "bbox": [round(v) for v in box], "score": 0.9})
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        out["text"].append({"text": "34 ABC 123", "bbox": [cx - 20, cy - 10, cx + 20, cy + 10]})
        return out


@pytest.mark.parametrize("move,expected", [(+1, "coming"), (-1, "going")])
def test_end_to_end_on_a_video(tmp_path, monkeypatch, move, expected):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(20):
        frame = np.zeros((240, 320, 3), np.uint8)
        size = 40 + 2 * i if move > 0 else 80 - 2 * i
        y = 20 + 6 * i if move > 0 else 140 - 6 * i
        cv2.rectangle(frame, (140 - size // 2, y), (140 + size // 2, y + size // 2), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    monkeypatch.setitem(sys.modules, "cactus", types.SimpleNamespace(Mirage=FakeMirage))
    monkeypatch.setattr(tm, "CROP_MIN_WIDTH", 10)
    out = tmp_path / "out"
    tm.main(["--source", str(video), "--no-window", "--output", str(out), "--max-fps", "0"])
    rows = list(csv.DictReader(open(out / "events.csv", encoding="utf-8")))
    assert len(rows) == 1, rows
    assert rows[0]["direction"] == expected
    assert rows[0]["plate"] == "34ABC123" and rows[0]["plate_confirmed"] == "True"
    assert (out / rows[0]["snapshot"]).exists()


# Frames 1 and 2 as Mirage returned them from a real camera (0-1000 grid).
REAL_FRAME_1 = {
    "objects": [
        {"id": 1, "name": "tree", "bbox": [64, 122, 236, 195]},
        {"id": 2, "name": "car", "bbox": [315, 210, 385, 265]},
        {"id": 3, "name": "car", "bbox": [117, 208, 394, 443]},
        {"id": 4, "name": "car", "bbox": [225, 221, 382, 426]},
        {"id": 5, "name": "car", "bbox": [460, 131, 928, 436]},
        {"id": 6, "name": "wooden pallet", "bbox": [11, 475, 406, 796]},
        {"id": 8, "name": "text", "bbox": [737, 797, 835, 813]},
    ],
    "text": [{"string": "Camera01", "bbox": [731, 791, 841, 818]}],
}
REAL_FRAME_2 = {
    "objects": [
        {"id": 4, "name": "car", "bbox": [110, 220, 376, 508]},
        {"id": 6, "name": "car", "bbox": [323, 132, 938, 766]},
        {"id": 7, "name": "license plate", "bbox": [379, 619, 516, 693]},
        {"id": 8, "name": "grille", "bbox": [333, 453, 619, 730]},
        {"id": 9, "name": "watermark", "bbox": [734, 798, 837, 813]},
    ],
    "text": [{"string": "BSLIZ6B", "bbox": [374, 607, 521, 701]},
             {"string": "Camera01", "bbox": [726, 791, 845, 818]}],
}


def test_vehicle_words_are_whole_words():
    assert tm.is_vehicle("car") and tm.is_vehicle("pickup truck") and tm.is_vehicle("cars")
    for label in ("bush", "scar", "card", "license plate", "car door", "grille", "caravan park"):
        assert not tm.is_vehicle(label), label


def test_real_frame_duplicates_dropped_small_car_behind_kept():
    vehicles, plates, ignored = tm.split_objects(tm.parse_objects(REAL_FRAME_1, 1000, 1000))
    boxes = sorted(v[2] for v in vehicles)
    # car 4 is a second box on car 3; car 2 is a distant car and stays.
    assert boxes == [(117, 208, 394, 443), (315, 210, 385, 265), (460, 131, 928, 436)]
    assert plates == [] and ignored == []


def test_camera_name_is_never_a_plate():
    assert tm.plate_candidates("Camera01") == []
    tracker = tm.Tracker()
    active, _ = tracker.update([("car", None, (0, 0, 1000, 1000))], 1000, 1000, 0)
    tm.assign_text(active, tm.parse_text(REAL_FRAME_1, 1000, 1000))
    assert not active[0].plate_votes


def test_real_frame_plate_box_and_uk_layout(monkeypatch):
    monkeypatch.setattr(tm, "PLATE_FORMATS", ("AA99AAA",))
    monkeypatch.setattr(tm, "IGNORE_TEXT", ())   # rely on the "watermark" box alone
    vehicles, plates, ignored = tm.split_objects(tm.parse_objects(REAL_FRAME_2, 1000, 1000))
    assert plates == [(379, 619, 516, 693)] and ignored == [(734, 798, 837, 813)]
    active, _ = tm.Tracker().update(vehicles, 1000, 1000, 0)
    tm.assign_plates(active, plates)
    tm.assign_text(active, tm.parse_text(REAL_FRAME_2, 1000, 1000), ignored)
    close_car = next(t for t in active if t.box == (323, 132, 938, 766))
    assert close_car.plate_box == (379, 619, 516, 693)
    assert dict(close_car.plate_votes) == {"BS11ZGB": tm.PLATE_BOX_VOTES}
    assert not any(t.plate_votes for t in active if t is not close_car)


def test_plate_layouts(monkeypatch):
    monkeypatch.setattr(tm, "PLATE_FORMATS", ("AA99AAA",))
    assert tm.plate_candidates("BSLIZ6B") == ["BS11ZGB"]
    assert tm.plate_candidates("GB BS11 ZGB") == ["BS11ZGB"]   # country band dropped
    assert tm.plate_candidates("STOP") == []
    assert tm.plate_candidates("SPEED 50") == []
    monkeypatch.setattr(tm, "PLATE_FORMATS", ("99A9999", "99AA999", "99AAA99"))
    assert tm.plate_candidates("34 ABC 12") == ["34ABC12"]
    assert tm.plate_candidates("B4 AB 123") == ["84AB123"]


def test_plate_format_argument(monkeypatch):
    monkeypatch.setattr(tm, "PLATE_FORMATS", ())
    tm.parse_args(["--plate-format", "aa99aaa", "--no-window"])
    assert tm.PLATE_FORMATS == ("AA99AAA",)
    with pytest.raises(SystemExit):
        tm.parse_args(["--plate-format", "AA-99"])


def test_gui_settings_parsing():
    pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    import traffic_gui as gui
    assert gui.parse_value("int", (1, 10), " 4 ") == 4
    assert gui.parse_value("optfloat", (0, 1, 0.1), "") is None
    assert gui.parse_value("layouts", None, "aa99aaa, 99A9999") == ("AA99AAA", "99A9999")
    assert gui.parse_value("list", None, "car, , truck") == ("car", "truck")
    for kind, extra, text in [("int", (1, 10), "11"), ("int", (1, 10), "x"), ("float", (0, 1, 0.1), "2"),
                              ("layouts", None, "AA-99"), ("choice", ("up", "down"), "left")]:
        with pytest.raises(ValueError):
            gui.parse_value(kind, extra, text)
    with pytest.raises(Exception):
        gui.parse_value("regex", None, "([")
    # Every control's current value survives a round trip through its text.
    for name, kind, _, _, extra in gui.all_specs():
        if kind != "bool":
            value = getattr(tm, name)
            assert gui.parse_value(kind, extra, gui.to_text(kind, value)) == value, name


def test_gui_settings_file_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    import traffic_gui as gui
    monkeypatch.setattr(gui, "SETTINGS_FILE", tmp_path / "s.json")
    for name, value in {"APPROACH": "up", "PLATE_FORMATS": ("AA99AAA",), "CROP_PASS": False,
                        "OBJECT_CONF": 0.4, "SOURCE": "rtsp://x"}.items():
        monkeypatch.setattr(tm, name, value)
    gui.save_settings()
    for name in ("APPROACH", "PLATE_FORMATS", "CROP_PASS", "OBJECT_CONF", "SOURCE"):
        monkeypatch.setattr(tm, name, gui.DEFAULTS[name])
    (tmp_path / "s.json").write_text((tmp_path / "s.json").read_text().replace('"MAX_FPS": 20.0', '"MAX_FPS": "bad"'))
    gui.load_settings()
    assert (tm.APPROACH, tm.PLATE_FORMATS, tm.CROP_PASS, tm.OBJECT_CONF, tm.SOURCE) == \
        ("up", ("AA99AAA",), False, 0.4, "rtsp://x")
    assert tm.MAX_FPS == gui.DEFAULTS["MAX_FPS"]   # a bad saved value is ignored


def test_missing_package_message_names_pip_package_and_python():
    message = tm.missing_message("cactus")
    assert "'cactus-compute'" in message and sys.executable in message
    assert "'huggingface_hub'" in tm.missing_message("huggingface_hub")


def test_mirage_loads_from_package_or_this_clone():
    assert tm.load_mirage().__name__ == "Mirage"


# ------------------------------------------------------------ motion gate --

def street(cars=(), brightness=80, size=(360, 640)):
    frame = np.full((*size, 3), brightness, np.uint8)
    for x0, y0, x1, y1 in cars:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return frame


def test_motion_gate_skips_still_frames_and_crops_motion(monkeypatch):
    monkeypatch.setattr(tm, "MOTION_LEARN_SECONDS", 10.0)
    gate = tm.MotionGate()
    parked = (500, 250, 600, 320)
    assert gate.update(street([parked]), 0.0).action == "skip"      # first frame: learn it, nothing moves yet
    assert gate.update(street([parked]), 0.1).action == "skip"      # parked car is background
    moving = (100, 100, 180, 160)
    motion = gate.update(street([parked, moving]), 0.2)
    assert motion.action == "crop"
    r = motion.region
    assert r[0] <= 100 and r[1] <= 100 and r[2] >= 180 and r[3] >= 160
    assert tm.area(r) < 0.6 * 640 * 360 and not tm.contains_point(r, 550, 285)


def test_motion_gate_absorbs_parked_cars_but_not_protected_ones(monkeypatch):
    monkeypatch.setattr(tm, "MOTION_LEARN_SECONDS", 1.0)
    car = (100, 100, 180, 160)

    def actions(protect):
        gate = tm.MotionGate()
        gate.update(street(), 0.0)
        return [gate.update(street([car]), t / 2, protect).action for t in range(1, 13)]

    parked = actions(())
    assert parked[0] == "crop" and parked[-1] == "skip"         # a car that stays fades into the background
    assert set(actions((car,))) == {"crop"}                      # a tracked, not-parked car never does


def test_motion_gate_resets_on_whole_scene_change_and_ignores_clock(monkeypatch):
    gate = tm.MotionGate()
    gate.update(street(), 0.0)
    changed = gate.update(street(brightness=200), 0.1)
    assert changed.action == "skip" and "reset" in changed.reason
    assert gate.update(street(brightness=200), 0.2).action == "skip"
    gate.ignore = [(520, 320, 640, 360)]                       # e.g. a "watermark" clock
    clock = street(brightness=200)
    clock[330:350, 540:620] = 0
    assert gate.update(clock, 0.3).action == "skip"


def test_tracks_outside_the_analysed_area_end_by_time(monkeypatch):
    monkeypatch.setattr(tm, "MAX_MISSED", 2)
    monkeypatch.setattr(tm, "MAX_UNSEEN_SECONDS", 2.0)
    tracker = tm.Tracker()
    tracker.update([("car", None, (10, 10, 50, 50))], 100, 100, 0.0)
    for t in (0.5, 1.0, 1.5):    # Mirage looked elsewhere: no missed frames counted
        assert tracker.update([], 100, 100, t, region=(60, 60, 100, 100)) == ([], [])
    assert tracker.update([], 100, 100, 1.9, region=False) == ([], [])
    _, finished = tracker.update([], 100, 100, 2.5, region=False)
    assert len(finished) == 1


def test_parked_vehicles_are_flagged_and_not_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "PARKED_SECONDS", 5.0)
    tracker = tm.Tracker()
    for t in range(8):
        active, _ = tracker.update([("car", None, (10 + t % 2, 10, 50, 50))], 100, 100, float(t))
    parked = active[0]
    assert parked.is_parked(7.0) and not parked.ever_moved
    log = tm.EventLog(str(tmp_path))
    assert log.write(parked) is None
    monkeypatch.setattr(tm, "LOG_PARKED", True)
    assert log.write(parked) is not None
    log.close()


class CountingMirage:
    """Finds every white rectangle; counts calls and records image sizes."""
    calls = []

    def __init__(self, weights=None):
        CountingMirage.calls = []

    def describe(self, image, *, sections, **_):
        CountingMirage.calls.append((image.shape[1], image.shape[0], tuple(sections)))
        h, w = image.shape[:2]
        n, _, stats, _ = cv2.connectedComponentsWithStats((image[:, :, 0] > 200).astype(np.uint8))
        out = {"objects": [], "text": []}
        for k in range(1, n):
            x, y, bw, bh, a = stats[k]
            if a < 300:
                continue
            box = [x / w * 1000, y / h * 1000, (x + bw) / w * 1000, (y + bh) / h * 1000]
            if "objects" in sections:
                out["objects"].append({"id": k, "name": "car", "bbox": [round(v) for v in box]})
            out["text"].append({"string": "34 ABC 123", "bbox": [round(v) for v in box]})
        return out


def test_detections_cut_by_the_crop_border_are_dropped():
    region = (100, 50, 300, 250)
    assert tm.cut_by_crop((280, 100, 299, 150), region, 640, 360)       # sliced at the right edge
    assert not tm.cut_by_crop((150, 100, 250, 200), region, 640, 360)   # wholly inside
    assert not tm.cut_by_crop((0, 100, 50, 150), (0, 50, 300, 250), 640, 360)  # frame edge is real


@pytest.mark.parametrize("parked", [(480, 240, 600, 320), (200, 150, 300, 210)],
                         ids=["parked-far", "parked-next-to-the-road"])
def test_end_to_end_motion_gate_saves_calls_and_hides_parked_car(tmp_path, monkeypatch, parked):
    video = tmp_path / "street.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for i in range(60):
        cars = [parked]
        if 15 <= i < 40:                      # one car drives down the left side
            y = 20 + 9 * (i - 15)
            cars.append((80, y, 160 + (i - 15), y + 60))
        writer.write(street(cars))
    writer.release()
    monkeypatch.setitem(sys.modules, "cactus", types.SimpleNamespace(Mirage=CountingMirage))
    monkeypatch.setattr(tm, "CROP_PASS", False)
    out = tmp_path / "out"
    tm.main(["--source", str(video), "--no-window", "--output", str(out), "--max-fps", "0"])
    rows = list(csv.DictReader(open(out / "events.csv", encoding="utf-8")))
    assert [r["direction"] for r in rows] == ["coming"], rows     # parked car is not logged
    assert rows[0]["plate"] == "34ABC123"
    calls = CountingMirage.calls
    assert len(calls) <= 30, len(calls)               # 60 frames, Mirage only while the car moves
    assert any(w < 640 for w, _, _ in calls)          # and mostly on a crop, not the whole frame


def test_ghost_of_a_departed_car_fades_once_mirage_finds_nothing(monkeypatch):
    monkeypatch.setattr(tm, "MOTION_LEARN_SECONDS", 1000.0)   # slow learning: only absorb() helps
    gate = tm.MotionGate()
    gate.update(street([(100, 100, 180, 160)]), 0.0)            # learnt with a car that then leaves
    motion = gate.update(street(), 0.1)
    assert motion.action == "crop"                              # the ghost looks like motion
    for t in range(2, 6):
        gate.absorb(motion.boxes, 0.5)                          # Mirage found no vehicle there
        motion = gate.update(street(), t / 10)
    assert motion.action == "skip"


# --------------------------------------------------- German plates, ignore --

@pytest.mark.parametrize("read,plate", [
    ("SB-FN 79E", "SBFN79E"), ("M AB 1234", "MAB1234"), ("B A 1", "BA1"), ("HH AB 123H", "HHAB123H"),
    ("MÜ X 99", "MÜX99"), ("5B FN 79E", "SBFN79E"), ("SB FN 7SE", "SBFN75E"), ("SB FN 79", "SBFN79"),
])
def test_german_plates(monkeypatch, read, plate):
    monkeypatch.setattr(tm, "PLATE_REGEX", tm.GERMAN_PLATE_REGEX)
    assert tm.plate_candidates(read) == [plate]


# (Look-alike fixing can still turn a fragment such as "Z66E" into a valid
# German plate, ZG6E; votes across frames are what outweigh such reads.)
@pytest.mark.parametrize("read", ["ABC DE 12345", "34 ABC 123", "CAMERA01", "STOP", "MYPARKEN", "1234", "AB-CD"])
def test_not_german_plates(monkeypatch, read):
    monkeypatch.setattr(tm, "PLATE_REGEX", tm.GERMAN_PLATE_REGEX)
    assert tm.plate_candidates(read) == []


def test_german_is_the_default():
    import importlib
    fresh = importlib.reload(tm)
    assert fresh.PLATE_REGEX == fresh.GERMAN_PLATE_REGEX and fresh.MAX_FPS == 20.0


@pytest.mark.parametrize("read", ["Camera 01", "CAMERAO1", "CAMERA0I", "Camera01 10:42"])
def test_camera_name_lookalikes_are_ignored(read):
    assert tm.plate_candidates(read) == []


def test_brightness_change_of_the_whole_picture_is_not_motion(monkeypatch):
    scene = street()
    cv2.rectangle(scene, (300, 100), (500, 300), (150, 150, 150), -1)
    brighter = np.clip(scene.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    gate = tm.MotionGate()
    gate.update(scene, 0.0)
    assert gate.update(brighter, 0.1).action == "skip"
    monkeypatch.setattr(tm, "MOTION_EXPOSURE", False)
    gate = tm.MotionGate()
    gate.update(scene, 0.0)
    assert "reset" in gate.update(brighter, 0.1).reason        # without it: everything "moved"


class FakeLiveReader:
    """Stands in for an RTSP stream: 50 frames/s of a car driving down."""
    frames = 0

    def __init__(self, source, log=None):
        self.is_live = True
        self.n = 0

    def read(self, last_seq, timeout=0.5):
        time.sleep(0.02)
        self.n += 1
        i = min(self.n, 60)
        y = 20 + 4 * i
        return self.n, street([(100, y, 200, y + 60)])

    def close(self):
        pass


class SlowMirage(CountingMirage):
    def describe(self, image, **kw):
        time.sleep(0.08)                      # Mirage much slower than the stream
        return super().describe(image, **kw)


def test_live_video_runs_at_stream_rate_while_mirage_works_in_the_background(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "LatestFrameReader", FakeLiveReader)
    monkeypatch.setattr(tm, "MAX_FPS", 0)
    monkeypatch.setattr(tm, "CROP_PASS", False)
    shown = []

    def on_frame(frame, visible, stats, motion):
        shown.append((stats["mirage_frames"], [t.direction() for t in visible]))
        return len(shown) >= 60

    monitor = tm.Monitor(SlowMirage(), "rtsp://camera", str(tmp_path), on_frame=on_frame, log=lambda m: None)
    monitor.run()
    calls = len(SlowMirage.calls)
    assert len(shown) == 60 and 3 <= calls <= 30, calls    # video did not wait for Mirage
    assert any("coming" in dirs for _, dirs in shown)       # tracking still works from the frames Mirage saw
    rows = list(csv.DictReader(open(tmp_path / "events.csv", encoding="utf-8")))
    assert [r["direction"] for r in rows] == ["coming"]


# ------------------------------------------------- a Mirage slower than 1 s --

class CrawlingCarReader(FakeLiveReader):
    """50 frames/s; a car creeping down 1 pixel per frame (50 px/s)."""

    def read(self, last_seq, timeout=0.5):
        time.sleep(0.02)
        self.n += 1
        y = 20 + self.n
        return self.n, street([(100, y, 200, y + 60)])


class OneSecondMirage(CountingMirage):
    def describe(self, image, **kw):
        time.sleep(1.0)
        return super().describe(image, **kw)


def test_boxes_stay_on_screen_and_one_track_when_mirage_takes_a_second(tmp_path, monkeypatch):
    """The user's case: Mirage needs about a second per pass. Boxes used to
    vanish (only vehicles seen in the last second were drawn) and every pass
    started a new vehicle (the boxes no longer overlapped)."""
    monkeypatch.setattr(tm, "LatestFrameReader", CrawlingCarReader)
    monkeypatch.setattr(tm, "MAX_FPS", 0)
    monkeypatch.setattr(tm, "CROP_PASS", False)
    shown = []

    def on_frame(frame, visible, stats, motion):
        shown.append(([t.id for t in visible], stats["mirage_frames"], tm.mirage_status(stats)))
        return len(shown) >= 230

    tm.Monitor(OneSecondMirage(), "rtsp://camera", str(tmp_path), on_frame=on_frame, log=lambda m: None).run()
    after_first = [ids for ids, passes, _ in shown if passes >= 1]
    assert len(after_first) > 100
    assert sum(1 for ids in after_first if ids) >= 0.95 * len(after_first)   # boxes on screen
    assert {i for ids in after_first for i in ids} == {1}                       # one vehicle throughout
    assert "s per pass" in shown[-1][2]
    rows = list(csv.DictReader(open(tmp_path / "events.csv", encoding="utf-8")))
    assert [r["direction"] for r in rows] == ["coming"]


def test_tracker_matches_by_distance_and_speed():
    tracker = tm.Tracker()
    tracker.update([("car", None, (100, 100, 200, 160))], 1000, 1000, 0.0)
    tracker.update([("car", None, (100, 180, 200, 240))], 1000, 1000, 1.0)   # no overlap: matched by distance
    assert list(tracker.tracks) == [1] and tracker.tracks[1].velocity == (0.0, 80.0)
    tracker.update([("car", None, (100, 260, 200, 320))], 1000, 1000, 2.0)   # predicted position overlaps
    assert list(tracker.tracks) == [1]
    tracker.update([("car", None, (700, 700, 800, 760))], 1000, 1000, 3.0)   # far away: a different car
    assert sorted(tracker.tracks) == [1, 2]


def test_next_car_in_the_queue_is_not_taken_for_the_one_ahead():
    tracker = tm.Tracker()
    tracker.update([("car", None, (100, 300, 200, 360))], 1000, 1000, 0.0)
    tracker.update([("car", None, (100, 380, 200, 440))], 1000, 1000, 1.0)   # driving down at 80 px/s
    # Car A is not detected; car B appears behind it (further up), within reach by distance.
    tracker.update([("car", None, (100, 250, 200, 310))], 1000, 1000, 2.0)
    assert sorted(tracker.tracks) == [1, 2]


# ------------------------------------------------ masking and ignore zones --

def test_without_masking_first_frame_and_reset_go_to_mirage_whole(monkeypatch):
    monkeypatch.setattr(tm, "MASK_STATIC", False)
    gate = tm.MotionGate()
    assert gate.update(street(), 0.0).action == "full"
    assert gate.update(street(brightness=200), 0.1).action == "full"


def test_mirage_sees_only_moving_areas_and_never_ignore_zones(monkeypatch):
    parked = (450, 200, 560, 280)
    moving = (100, 100, 180, 160)
    frame = street([parked, moving])
    motion = tm.Motion("full", "test", boxes=[moving])
    image, offset, keep = tm.mirage_input(frame, motion)
    assert offset == (0, 0) and image.shape == frame.shape
    assert (image[220:260, 470:540] == tm.MASK_GREY).all()            # parked car painted out
    assert (image[110:150, 110:170] == 255).all()                     # moving car untouched
    assert tm.visible_share(parked, keep) == 0.0 and tm.visible_share(moving, keep) == 1.0
    monkeypatch.setattr(tm, "IGNORE_ZONES", ((0.1, 0.2, 0.4, 0.6),))   # covers the moving car
    image, _, keep = tm.mirage_input(frame, motion)
    assert (image[110:150, 110:170] == tm.MASK_GREY).all()
    crop = tm.Motion("crop", "test", region=(50, 50, 300, 250), boxes=[moving])
    monkeypatch.setattr(tm, "IGNORE_ZONES", ())
    image, offset, _ = tm.mirage_input(frame, crop)
    assert offset == (50, 50) and image.shape[:2] == (200, 250)


def test_motion_inside_an_ignore_zone_is_not_motion(monkeypatch):
    monkeypatch.setattr(tm, "IGNORE_ZONES", ((0.6, 0.5, 1.0, 1.0),))
    gate = tm.MotionGate()
    gate.update(street(), 0.0)
    assert gate.update(street([(450, 220, 600, 330)]), 0.1).action == "skip"   # inside the zone
    assert gate.update(street([(50, 50, 150, 120)]), 0.2).action == "crop"      # outside it


def test_mirage_view_drawing():
    view = tm.MirageView(3, street([(100, 100, 180, 160)]), (10, 20),
                         [("car", None, (100, 100, 180, 160)), ("tree", None, (10, 10, 60, 80))],
                         [("SB FN 79E", (110, 140, 170, 155))], 1.234, False)
    picture = tm.draw_mirage_view(view)
    assert picture.shape[0] == 360 + 26 and picture.shape[1] == 640


def test_end_to_end_parked_car_beside_the_road_is_never_tracked(tmp_path, monkeypatch):
    """A parked car right next to the road ends up inside the crop: masking
    paints it out, so Mirage never reports it and it never becomes a track."""
    video = tmp_path / "street.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    parked = (200, 150, 300, 210)      # 20 px from the road: a slice shows in the moving car's margin
    for i in range(50):
        cars = [parked]
        if 10 <= i < 40:
            y = 20 + 9 * (i - 10)
            cars.append((80, y, 180, y + 60))
        writer.write(street(cars))
    writer.release()
    monkeypatch.setitem(sys.modules, "cactus", types.SimpleNamespace(Mirage=CountingMirage))
    monkeypatch.setattr(tm, "CROP_PASS", False)
    monkeypatch.setattr(tm, "MOTION_MIN_CROP", 0.6)   # crop wider than the visible margin: grey cuts the slice
    seen = []
    real = tm.Monitor._analyse

    def spy(self, frame, motion, now):
        real(self, frame, motion, now)
        seen.append(self.last_view)

    monkeypatch.setattr(tm.Monitor, "_analyse", spy)
    tracked = []
    real_update = tm.Tracker.update

    def update_spy(self, detections, *args, **kwargs):
        tracked.extend(d[2] for d in detections)
        return real_update(self, detections, *args, **kwargs)

    monkeypatch.setattr(tm.Tracker, "update", update_spy)
    out = tmp_path / "out"
    tm.main(["--source", str(video), "--no-window", "--output", str(out), "--max-fps", "0"])
    rows = list(csv.DictReader(open(out / "events.csv", encoding="utf-8")))
    assert [r["direction"] for r in rows] == ["coming"]
    # Not even a slice of the parked car (showing through the margin around the moving one) is tracked.
    assert tracked and not any(tm.area(tm.overlap(box, parked)) > 0 for box in tracked), tracked
    # Mirage never saw the whole parked car (it was grey apart from a slice in the margin).
    for view in seen:
        x0, y0 = view.offset
        for label, _, box in view.objects:
            assert tm.iou(tm.shift(box, x0, y0), parked) < 0.5, (view.number, box)


def test_slice_of_a_still_object_at_the_edge_of_the_visible_area_is_dropped():
    keep = np.zeros((360, 640), bool)
    keep[50:250, 50:300] = True                                   # visible area around a moving car
    assert not tm.cut_by_mask((120, 100, 220, 160), keep)         # the moving car, well inside
    assert tm.cut_by_mask((270, 120, 300, 200), keep)             # slice of a parked car at the edge
    assert not tm.cut_by_mask((0, 0, 50, 50), None)


def test_a_big_still_object_does_not_capture_a_car_nearby():
    tracker = tm.Tracker()
    tracker.update([("car", None, (510, 430, 891, 539))], 960, 540, 0.0)    # big white thing, wrongly "car"
    tracker.update([("car", None, (200, 134, 341, 225))], 960, 540, 1.0)    # a real car, 1.4 big-boxes away
    assert sorted(tracker.tracks) == [1, 2]


def test_without_plates_mirage_is_asked_for_objects_only(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "READ_PLATES", False)
    video = tmp_path / "street.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for i in range(40):
        y = 20 + 7 * i
        writer.write(street([(80, y, 180, y + 60)]))
    writer.release()
    monkeypatch.setitem(sys.modules, "cactus", types.SimpleNamespace(Mirage=CountingMirage))
    out = tmp_path / "out"
    tm.main(["--source", str(video), "--no-window", "--output", str(out), "--max-fps", "0"])
    calls = CountingMirage.calls
    assert calls and all(sections == ("objects",) for _, _, sections in calls)   # no text, no plate reads
    rows = list(csv.DictReader(open(out / "events.csv", encoding="utf-8")))
    assert [(r["direction"], r["plate"]) for r in rows] == [("coming", "")]


def test_plates_default_off_and_switch():
    import importlib
    fresh = importlib.reload(tm)
    assert fresh.READ_PLATES is False
    fresh.parse_args(["--plates", "--no-window"])
    assert fresh.READ_PLATES is True


def test_a_slow_video_file_is_not_mistaken_for_its_end(tmp_path, monkeypatch):
    """Decoding pauses longer than the read timeout must not end a recording early."""
    class SlowFile:
        def __init__(self, source, log=None):
            self.is_live, self.n, self.ended = False, 0, False

        def read(self, last_seq, timeout=0.5):
            self.n += 1
            if self.n % 3 == 0:
                return last_seq, None          # timed out: decoder busy, not the end
            if self.n > 90:
                self.ended = True
                return last_seq, None
            y = 20 + 4 * self.n
            return self.n, street([(80, y, 180, y + 60)])

        def close(self):
            pass

    monkeypatch.setattr(tm, "LatestFrameReader", SlowFile)
    monkeypatch.setattr(tm, "MAX_FPS", 0)
    monkeypatch.setattr(tm, "READ_PLATES", False)
    seen = []
    tm.Monitor(CountingMirage(), "clip.avi", str(tmp_path), on_frame=lambda f, v, s, m: seen.append(1) and False,
               log=lambda m: None).run()
    assert len(seen) == 60                      # every real frame, not just those before the first pause


def test_mirage_view_of_a_small_crop_is_padded_for_the_caption():
    view = tm.MirageView(1, street(size=(160, 280)), (0, 0), [], [], 0.5, False)
    assert tm.draw_mirage_view(view).shape[1] == 560
