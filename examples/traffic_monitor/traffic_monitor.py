"""Traffic monitor: read an RTSP camera, detect vehicles with Mirage, read
their number plates, and work out whether each vehicle is coming or going.

How it works
------------
1. A background thread reads the RTSP stream and keeps only the newest frame,
   so a slow model never builds up a delay behind live video.
2. Each processed frame goes through Mirage (``sections=["objects", "text"]``).
   Objects whose label looks like a vehicle are kept.
3. A small IoU tracker links vehicle boxes across frames into tracks.
4. Direction comes from how a track moves: towards the bottom of the frame
   and growing means approaching the camera, the reverse means leaving.
   ``APPROACH`` says which screen direction counts as "coming".
5. Plates: text that Mirage reads inside a vehicle box is a candidate. For
   vehicles without a confirmed plate, the vehicle is also cropped from the
   full-resolution frame and read again, because Mirage downsizes its input
   and plates are small. Candidates are cleaned, checked against
   ``PLATE_REGEX`` and voted on across frames.
6. When a track disappears, one row is written to ``traffic_output/events.csv``
   together with a snapshot of the vehicle.

Edit the SETTINGS block below, or override it with command-line arguments
(in Visual Studio: Project > Properties > Debug > Script Arguments).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- SETTINGS --

# RTSP URL, a video file path, or a webcam number such as "0".
SOURCE = "rtsp://127.0.0.1:8554/Streaming/Channels/101"

# Which way a vehicle moves on screen when it is COMING towards the camera:
# "down", "up", "left" or "right".
APPROACH = "down"

# Print the raw JSON Mirage returns for the first few frames. Turn this on
# for your first run to check the key names this parser looks for.
DUMP_JSON = False
DUMP_JSON_FRAMES = 3

# Mirage boxes are on a 0-1000 grid. If boxes are drawn in the wrong place,
# switch the order to "yxyx".
BBOX_ORDER = "xyxy"
BBOX_SCALE = 1000.0

# Object labels that count as vehicles (matched as substrings, lowercase).
VEHICLE_WORDS = ("car", "truck", "bus", "van", "vehicle", "motorcycle", "motorbike",
                 "suv", "taxi", "pickup", "lorry", "jeep", "minivan")

# Plate format after cleaning (uppercase letters and digits only). The default
# accepts 4 to 9 characters with at least one digit and one letter. Tighten it
# to your country's format, e.g. UK: r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$",
# Turkey: r"^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$".
PLATE_REGEX = r"^(?=.*[0-9])(?=.*[A-Z])[A-Z0-9]{4,9}$"
PLATE_CONFIRM_VOTES = 3        # identical reads needed before a plate is final

# Extra plate pass on a full-resolution crop of each unconfirmed vehicle.
CROP_PASS = True
CROP_EVERY = 2                 # only every Nth processed frame
CROP_MIN_WIDTH = 120           # skip vehicles narrower than this (pixels)
CROP_PAD = 0.06                # grow the crop by this fraction on each side

# Confidence thresholds handed to Mirage (None = model default).
OBJECT_CONF = None
TEXT_CONF = None

# Tracker.
IOU_MATCH = 0.25               # minimum overlap to continue a track
MAX_MISSED = 8                 # processed frames a track may go unseen
MIN_TRACK_FRAMES = 3           # shorter tracks are dropped as noise
DIRECTION_THRESHOLD = 0.08     # how much movement is needed to call a direction

MAX_FPS = 10.0                 # cap on processed frames per second
SHOW_WINDOW = True             # live preview window (press Q or Esc to stop)
OUTPUT_DIR = "traffic_output"
SAVE_SNAPSHOTS = True

# ----------------------------------------------------------- END SETTINGS --

# Prefer TCP for RTSP: UDP drops packets on busy networks and smears frames.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402  (must come after the environment variable above)
import numpy as np  # noqa: E402


# ------------------------------------------------------------ video input --

class LatestFrameReader:
    """Reads a stream on a thread and always hands out the newest frame."""

    def __init__(self, source: str):
        self.source = int(source) if source.isdigit() else source
        self.is_live = isinstance(self.source, int) or "://" in str(self.source)
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ended = threading.Event()
        self._cap = self._open()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG) if isinstance(self.source, str) \
            else cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit(f"could not open video source: {redact(str(self.source))}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if not self.is_live:
                    self._ended.set()
                    return
                print("stream dropped, reconnecting in 2 s ...", file=sys.stderr)
                self._cap.release()
                time.sleep(2)
                try:
                    self._cap = self._open()
                except SystemExit as exc:
                    print(exc, file=sys.stderr)
                continue
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1
            if not self.is_live:
                # Files decode faster than real time; wait for the consumer so
                # no frames are skipped when testing with a recording.
                while not self._stop.is_set() and self._frame is not None:
                    time.sleep(0.001)

    def read(self, last_seq: int, timeout: float = 5.0):
        """Return (seq, frame) newer than last_seq, or (last_seq, None) at the end."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    frame, seq = self._frame, self._seq
                    if not self.is_live:
                        self._frame = None
                    return seq, frame
            if self._ended.is_set():
                return last_seq, None
            time.sleep(0.005)
        return last_seq, None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


def redact(url: str) -> str:
    """Hide the password in rtsp://user:password@host URLs before printing."""
    return re.sub(r"(://[^:/@]+):[^/]*@", r"\1:***@", url)


# ---------------------------------------------------------- Mirage parsing --

LABEL_KEYS = ("label", "name", "class", "category", "type", "id")
BOX_KEYS = ("bbox", "box", "boxes", "bboxes")
SCORE_KEYS = ("score", "conf", "confidence", "probability")
TEXT_KEYS = ("text", "string", "value", "content", "line")


def _first(obj: dict, keys):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def _boxes(value):
    """Yield every 4-number box in value (one box or a list of boxes)."""
    if isinstance(value, dict):
        if all(k in value for k in ("x0", "y0", "x1", "y1")):
            yield [value["x0"], value["y0"], value["x1"], value["y1"]]
        elif all(k in value for k in ("x", "y", "w", "h")):
            yield [value["x"], value["y"], value["x"] + value["w"], value["y"] + value["h"]]
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            yield list(value)
        else:
            for item in value:
                yield from _boxes(item)


def to_pixels(box, width: int, height: int):
    """Convert a Mirage box to integer pixel coordinates (x0, y0, x1, y1)."""
    a, b, c, d = (float(v) for v in box)
    if BBOX_ORDER == "yxyx":
        a, b, c, d = b, a, d, c
    scale = 1.0 if max(a, b, c, d) <= 1.0 else BBOX_SCALE
    x0, x1 = sorted((a / scale * width, c / scale * width))
    y0, y1 = sorted((b / scale * height, d / scale * height))
    clamp = lambda v, hi: int(max(0, min(hi, round(v))))  # noqa: E731
    return clamp(x0, width - 1), clamp(y0, height - 1), clamp(x1, width - 1), clamp(y1, height - 1)


def _items(result: dict, section: str):
    value = result.get(section) if isinstance(result, dict) else None
    if isinstance(value, dict):  # some versions may nest e.g. {"lines": [...]}
        value = next((v for v in value.values() if isinstance(v, list)), [])
    return value if isinstance(value, list) else []


def parse_objects(result: dict, width: int, height: int):
    """Return [(label, score, (x0, y0, x1, y1)), ...] for every object box."""
    found = []
    for obj in _items(result, "objects"):
        if not isinstance(obj, dict):
            continue
        label = str(_first(obj, LABEL_KEYS) or "").lower()
        score = _first(obj, SCORE_KEYS)
        score = float(score) if isinstance(score, (int, float)) else None
        for box in _boxes(_first(obj, BOX_KEYS)):
            px = to_pixels(box, width, height)
            if px[2] - px[0] >= 4 and px[3] - px[1] >= 4:
                found.append((label, score, px))
    return found


def parse_text(result: dict, width: int, height: int):
    """Return [(text, box_or_None), ...] for every text line."""
    found = []
    for line in _items(result, "text"):
        if isinstance(line, str):
            found.append((line, None))
            continue
        if not isinstance(line, dict):
            continue
        text = _first(line, TEXT_KEYS)
        if not isinstance(text, str):
            continue
        boxes = list(_boxes(_first(line, BOX_KEYS)))
        found.append((text, to_pixels(boxes[0], width, height) if boxes else None))
    return found


def is_vehicle(label: str) -> bool:
    return any(word in label for word in VEHICLE_WORDS)


# ------------------------------------------------------------------ plates --

_plate_re = re.compile(PLATE_REGEX)
# Characters OCR commonly swaps; tried when a read does not fit the format.
_AMBIGUOUS = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S",
              "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G"}


def clean_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def plate_candidates(text: str):
    """Plates that text may contain, best guess first."""
    cleaned = clean_plate(text)
    if not cleaned:
        return []
    if _plate_re.match(cleaned):
        return [cleaned]
    # Try fixing one ambiguous character at a time, but only for text that
    # already mixes letters and digits, so signs such as "STOP" stay out.
    if not (re.search(r"[0-9]", cleaned) and re.search(r"[A-Z]", cleaned)):
        return []
    for i, ch in enumerate(cleaned):
        if ch in _AMBIGUOUS:
            fixed = cleaned[:i] + _AMBIGUOUS[ch] + cleaned[i + 1:]
            if _plate_re.match(fixed):
                return [fixed]
    return []


# ----------------------------------------------------------------- tracker --

def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    label: str
    box: tuple
    first_seen: float
    last_seen: float
    history: list = field(default_factory=list)   # (cx, cy, area) normalised
    plate_votes: collections.Counter = field(default_factory=collections.Counter)
    missed: int = 0
    best_snapshot: object = None
    best_area: float = 0.0

    @property
    def plate(self):
        if not self.plate_votes:
            return None
        return self.plate_votes.most_common(1)[0][0]

    @property
    def plate_confirmed(self) -> bool:
        return bool(self.plate_votes) and self.plate_votes.most_common(1)[0][1] >= PLATE_CONFIRM_VOTES

    def direction(self) -> str:
        """'coming', 'going' or 'unknown' from movement over the track's life."""
        if len(self.history) < 2:
            return "unknown"
        (x0, y0, a0), (x1, y1, a1) = self.history[0], self.history[-1]
        step = {"down": y1 - y0, "up": y0 - y1, "right": x1 - x0, "left": x0 - x1}[APPROACH]
        # Growing boxes mean approaching; log keeps it symmetric (x2 vs /2).
        growth = math.log(a1 / a0) if a0 > 0 and a1 > 0 else 0.0
        score = step + 0.15 * growth
        if score > DIRECTION_THRESHOLD:
            return "coming"
        if score < -DIRECTION_THRESHOLD:
            return "going"
        return "unknown"


class Tracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections, width: int, height: int, now: float):
        """Match detections to tracks. Returns (matched_tracks, finished_tracks)."""
        pairs = sorted(((iou(t.box, d[2]), tid, di)
                        for tid, t in self.tracks.items() for di, d in enumerate(detections)),
                       reverse=True)
        used_tracks, used_dets, matched = set(), set(), []
        for overlap, tid, di in pairs:
            if overlap < IOU_MATCH:
                break
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matched.append((self.tracks[tid], detections[di]))
        for di, det in enumerate(detections):
            if di not in used_dets:
                track = Track(self._next_id, det[0], det[2], now, now)
                self._next_id += 1
                self.tracks[track.id] = track
                matched.append((track, det))
        for track, (label, _, box) in matched:
            track.box, track.label, track.last_seen, track.missed = box, label, now, 0
            cx = (box[0] + box[2]) / 2 / width
            cy = (box[1] + box[3]) / 2 / height
            area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
            track.history.append((cx, cy, area))
        finished = []
        live = {t.id for t, _ in matched}
        for tid in list(self.tracks):
            if tid not in live:
                track = self.tracks[tid]
                track.missed += 1
                if track.missed > MAX_MISSED:
                    finished.append(self.tracks.pop(tid))
        return [t for t, _ in matched], finished

    def flush(self):
        finished = list(self.tracks.values())
        self.tracks.clear()
        return finished


# ----------------------------------------------------------------- outputs --

class EventLog:
    FIELDS = ["track_id", "first_seen", "last_seen", "vehicle", "direction",
              "plate", "plate_votes", "plate_confirmed", "frames", "snapshot"]

    def __init__(self, directory: str):
        self.dir = Path(directory)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.csv"
        new = not self.path.exists()
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if new:
            self._csv.writeheader()

    def write(self, track: Track):
        if len(track.history) < MIN_TRACK_FRAMES:
            return
        stamp = datetime.fromtimestamp(track.first_seen)
        snapshot = ""
        if SAVE_SNAPSHOTS and track.best_snapshot is not None:
            name = f"{stamp:%Y%m%d_%H%M%S}_track{track.id}_{track.plate or 'noplate'}.jpg"
            cv2.imwrite(str(self.dir / "snapshots" / name), track.best_snapshot)
            snapshot = f"snapshots/{name}"
        top = track.plate_votes.most_common(1)
        row = {
            "track_id": track.id,
            "first_seen": stamp.isoformat(timespec="seconds"),
            "last_seen": datetime.fromtimestamp(track.last_seen).isoformat(timespec="seconds"),
            "vehicle": track.label,
            "direction": track.direction(),
            "plate": track.plate or "",
            "plate_votes": top[0][1] if top else 0,
            "plate_confirmed": track.plate_confirmed,
            "frames": len(track.history),
            "snapshot": snapshot,
        }
        self._csv.writerow(row)
        self._file.flush()
        print(f"[event] #{row['track_id']} {row['vehicle']:<10} {row['direction']:<8} "
              f"plate={row['plate'] or '-'} ({row['plate_votes']} votes)")

    def close(self):
        self._file.close()


def crop(frame, box, pad: float):
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = max(0, int(box[0] - bw * pad))
    y0 = max(0, int(box[1] - bh * pad))
    x1 = min(w, int(box[2] + bw * pad))
    y1 = min(h, int(box[3] + bh * pad))
    return frame[y0:y1, x0:x1]


def draw(frame, tracks, fps: float):
    colours = {"coming": (0, 200, 0), "going": (0, 140, 255), "unknown": (200, 200, 200)}
    for t in tracks:
        direction = t.direction()
        colour = colours[direction]
        x0, y0, x1, y1 = t.box
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        plate = t.plate or "?"
        if t.plate and not t.plate_confirmed:
            plate += "?"
        text = f"#{t.id} {t.label} {direction} {plate}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, max(0, y0 - th - 8)), (x0 + tw + 6, y0), colour, -1)
        cv2.putText(frame, text, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"{fps:.1f} fps  coming = {APPROACH}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# -------------------------------------------------------------------- main --

def mirage_options(sections):
    options = {"sections": sections, "fields": ["bbox"]}
    if OBJECT_CONF is not None:
        options["conf"] = OBJECT_CONF
    if TEXT_CONF is not None:
        options["text_conf"] = TEXT_CONF
    return options


def assign_text(tracks, lines):
    """Vote text lines whose centre falls inside a vehicle box for that vehicle."""
    for text, box in lines:
        if box is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        owners = [t for t in tracks if t.box[0] <= cx <= t.box[2] and t.box[1] <= cy <= t.box[3]]
        if not owners:
            continue
        # The smallest enclosing box is the most specific vehicle.
        owner = min(owners, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        for plate in plate_candidates(text):
            owner.plate_votes[plate] += 1


def parse_args(argv=None):
    global SOURCE, APPROACH, DUMP_JSON, BBOX_ORDER, SHOW_WINDOW, MAX_FPS, OUTPUT_DIR, PLATE_REGEX, _plate_re
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default=SOURCE, help="RTSP URL, video file or webcam number")
    p.add_argument("--approach", default=APPROACH, choices=["down", "up", "left", "right"],
                   help="screen direction a vehicle moves in when coming towards the camera")
    p.add_argument("--dump-json", action="store_true", default=DUMP_JSON,
                   help="print Mirage's raw JSON for the first frames")
    p.add_argument("--bbox-order", default=BBOX_ORDER, choices=["xyxy", "yxyx"])
    p.add_argument("--plate-regex", default=PLATE_REGEX)
    p.add_argument("--max-fps", type=float, default=MAX_FPS)
    p.add_argument("--no-window", action="store_true", help="run without the preview window")
    p.add_argument("--output", default=OUTPUT_DIR, help="folder for events.csv and snapshots")
    p.add_argument("--weights", default=None, help="path to mirage.cact (default: download)")
    args = p.parse_args(argv)
    SOURCE, APPROACH, DUMP_JSON = args.source, args.approach, args.dump_json
    BBOX_ORDER, MAX_FPS, OUTPUT_DIR = args.bbox_order, args.max_fps, args.output
    SHOW_WINDOW = SHOW_WINDOW and not args.no_window
    PLATE_REGEX, _plate_re = args.plate_regex, re.compile(args.plate_regex)
    return args


def main(argv=None):
    args = parse_args(argv)
    from cactus import Mirage

    print("loading Mirage (the first run downloads the engine and weights) ...")
    vision = Mirage(weights=args.weights)
    print(f"opening {redact(SOURCE)}")
    reader = LatestFrameReader(SOURCE)
    tracker, log = Tracker(), EventLog(OUTPUT_DIR)
    print(f"logging to {log.path.resolve()}")

    seq, processed, fps, window_ok = 0, 0, 0.0, SHOW_WINDOW
    min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
    try:
        while True:
            started = time.monotonic()
            seq, frame = reader.read(seq)
            if frame is None:
                if reader.is_live:
                    continue  # reconnecting; keep waiting
                print("end of video")
                break
            height, width = frame.shape[:2]
            rgb = np.ascontiguousarray(frame[:, :, ::-1])  # OpenCV gives BGR
            result = vision.describe(rgb, **mirage_options(["objects", "text"]))
            if DUMP_JSON and processed < DUMP_JSON_FRAMES:
                print(json.dumps(result, ensure_ascii=False, indent=2)[:6000])

            vehicles = [d for d in parse_objects(result, width, height) if is_vehicle(d[0])]
            now = time.time()
            active, finished = tracker.update(vehicles, width, height, now)
            assign_text(active, parse_text(result, width, height))

            for track in active:
                area = (track.box[2] - track.box[0]) * (track.box[3] - track.box[1])
                if SAVE_SNAPSHOTS and area > track.best_area:
                    track.best_area, track.best_snapshot = area, crop(frame, track.box, CROP_PAD).copy()
                if (CROP_PASS and not track.plate_confirmed and processed % max(1, CROP_EVERY) == 0
                        and track.box[2] - track.box[0] >= CROP_MIN_WIDTH):
                    piece = crop(frame, track.box, CROP_PAD)
                    piece_rgb = np.ascontiguousarray(piece[:, :, ::-1])
                    crop_result = vision.describe(piece_rgb, **mirage_options(["text"]))
                    for text, _ in parse_text(crop_result, piece.shape[1], piece.shape[0]):
                        for plate in plate_candidates(text):
                            track.plate_votes[plate] += 1

            for track in finished:
                log.write(track)

            processed += 1
            elapsed = time.monotonic() - started
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6)) if processed > 1 else 1.0 / max(elapsed, 1e-6)

            if window_ok:
                try:
                    cv2.imshow("Traffic monitor (Q to quit)", draw(frame.copy(), active, fps))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
                except cv2.error:
                    print("no display available; continuing without the preview window")
                    window_ok = False
            elif processed % 50 == 0:
                print(f"{processed} frames, {fps:.1f} fps, {len(tracker.tracks)} vehicles in view")

            if min_interval > elapsed:
                time.sleep(min_interval - elapsed)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        for track in tracker.flush():
            log.write(track)
        log.close()
        reader.close()
        if window_ok:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
