"""Offline tests for traffic_monitor.py.

Mirage is replaced by a fake that "sees" a white rectangle in a synthetic
video, so these run without the engine, the weights or a camera:

    python -m pytest examples/traffic_monitor
"""
import csv
import sys
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
    (tmp_path / "s.json").write_text((tmp_path / "s.json").read_text().replace('"MAX_FPS": 10.0', '"MAX_FPS": "bad"'))
    gui.load_settings()
    assert (tm.APPROACH, tm.PLATE_FORMATS, tm.CROP_PASS, tm.OBJECT_CONF, tm.SOURCE) == \
        ("up", ("AA99AAA",), False, 0.4, "rtsp://x")
    assert tm.MAX_FPS == gui.DEFAULTS["MAX_FPS"]   # a bad saved value is ignored
