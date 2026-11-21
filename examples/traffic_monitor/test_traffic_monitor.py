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
    monkeypatch.setattr(tm, "_plate_re", tm.re.compile(r"^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$"))
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
