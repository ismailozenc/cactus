"""Traffic monitor: read an RTSP camera, detect vehicles with Mirage, read
their number plates, and work out whether each vehicle is coming or going.

How it works
------------
1. A background thread reads the RTSP stream and keeps only the newest frame,
   so a slow model never builds up a delay behind live video.
2. A motion gate (``MotionGate``) compares a small grey copy of the frame
   with a learnt background image. Nothing moving: Mirage is skipped. Motion:
   only that part of the frame is sent, at full resolution, so parked cars
   are never looked at and moving plates get more pixels. Areas where Mirage
   then finds no vehicle fade into the background; vehicles being tracked
   never do until they are parked.
   Mirage reads the frame or crop (``sections=["objects", "text"]``).
   Objects whose label looks like a vehicle are kept, duplicate boxes of the
   same vehicle are dropped, and "license plate" objects are matched to the
   vehicle they sit on.
3. A small IoU tracker links vehicle boxes across frames into tracks.
4. Direction comes from how a track moves: towards the bottom of the frame
   and growing means approaching the camera, the reverse means leaving.
   ``APPROACH`` says which screen direction counts as "coming".
5. Plates: text that Mirage reads inside a vehicle box is a candidate, and
   text inside its plate box counts double. For vehicles without a confirmed
   plate, the plate box (or the whole vehicle) is also cropped from the
   full-resolution frame and read again, because Mirage downsizes its input
   and plates are small. The camera's own on-screen text is ignored.
   Candidates are cleaned, fitted to ``PLATE_FORMATS`` (or checked against
   ``PLATE_REGEX``) and voted on across frames.
6. When a track disappears, one row is written to ``traffic_output/events.csv``
   together with a snapshot of the vehicle.

Edit the SETTINGS block below, or override it with command-line arguments
(in Visual Studio: Project > Properties > Debug > Script Arguments).
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, replace
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

# Object labels that count as vehicles (whole words, lowercase). A label that
# also names a part ("car door", "license plate") is not a vehicle.
VEHICLE_WORDS = ("car", "truck", "bus", "van", "vehicle", "motorcycle", "motorbike",
                 "suv", "taxi", "pickup", "lorry", "jeep", "minivan")
VEHICLE_PART_WORDS = ("plate", "wheel", "tire", "tyre", "door", "window", "mirror", "light",
                      "headlight", "grille", "bumper", "seat", "hood", "roof", "logo", "badge")

# A vehicle box lying mostly inside a bigger one, and not much smaller, is
# a duplicate detection of the same vehicle and is dropped.
DUPLICATE_OVERLAP = 0.8        # share of the smaller box inside the bigger one
DUPLICATE_SIZE_RATIO = 0.4     # smaller/bigger area; tinier boxes are cars behind

# Mirage also detects number plates as objects. Their boxes are preferred for
# reading: text inside one gets PLATE_BOX_VOTES votes instead of 1, and the
# crop pass reads the plate box instead of the whole vehicle.
PLATE_WORDS = ("license plate", "licence plate", "number plate", "registration plate")
PLATE_BOX_VOTES = 2
PLATE_CROP_PAD = 0.25
PLATE_CROP_MIN_WIDTH = 24      # pixels; smaller plate boxes fall back to the vehicle crop

# Text that is never a plate, such as the camera's on-screen name. Compared
# after cleaning (uppercase, letters and digits only).
IGNORE_TEXT = ("CAMERA01",)
# Text inside objects with these names is skipped (Mirage calls the camera's
# on-screen text "watermark").
IGNORE_TEXT_IN = ("watermark", "timestamp", "logo")

# Plate layouts, A = letter and 9 = digit. When set, every read is fitted to
# them position by position, so for UK ("AA99AAA",) the read BSLIZ6B becomes
# BS11ZGB, and PLATE_REGEX is not used. Several layouts may be listed, e.g.
# Turkey: ("99A9999", "99A99999", "99AA999", "99AA9999", "99AAA99", "99AAA999").
PLATE_FORMATS = ()
PLATE_MAX_FIXES = 3            # most characters a layout may correct in one read

# Used when PLATE_FORMATS is empty: the plate after cleaning (uppercase,
# letters, digits and Ä Ö Ü only). Ready-made patterns:
#   GERMAN_PLATE_REGEX: district (1-3 letters), 1-2 letters, 1-4 digits not
#     starting with 0, then E (electric) or H (historic) optionally; at most
#     8 characters before the E/H. "SB-FN 79E" is read as SBFN79E.
#   ANY_PLATE_REGEX: 4 to 9 characters with at least one digit and one letter.
GERMAN_PLATE_REGEX = r"^(?=[A-ZÄÖÜ0-9]{3,8}[EH]?$)[A-ZÄÖÜ]{1,3}[A-Z]{1,2}[1-9][0-9]{0,3}[EH]?$"
ANY_PLATE_REGEX = r"^(?=.*[0-9])(?=.*[A-ZÄÖÜ])[A-ZÄÖÜ0-9]{4,9}$"
PLATE_REGEX = GERMAN_PLATE_REGEX
PLATE_CONFIRM_VOTES = 3        # identical reads needed before a plate is final

# Extra plate pass on a full-resolution crop of each unconfirmed vehicle.
CROP_PASS = True
CROP_EVERY = 1                 # only on every Nth Mirage pass (one vehicle per pass)
CROP_MIN_WIDTH = 120           # skip vehicles narrower than this (pixels)
CROP_PAD = 0.06                # grow the crop by this fraction on each side

# Confidence thresholds handed to Mirage (None = model default).
OBJECT_CONF = None
TEXT_CONF = None

# Tracker.
IOU_MATCH = 0.25               # minimum overlap to continue a track
MAX_MISSED = 8                 # analysed frames a track may go unseen
MAX_UNSEEN_SECONDS = 2.0       # a track outside the analysed area ends after this
MIN_TRACK_FRAMES = 3           # shorter tracks are dropped as noise
DIRECTION_THRESHOLD = 0.08     # how much movement is needed to call a direction

# Motion gate: a background image is learnt, and Mirage only looks where the
# frame differs from it. With no motion Mirage is skipped entirely; with
# motion only that part of the frame is sent, at full resolution.
MOTION_GATE = True
SHOW_MOTION = False            # tint moving pixels red and outline what Mirage reads
MOTION_THRESHOLD = 25          # grey-level change (0-255) that counts as motion
MOTION_MIN_AREA = 0.004        # smallest moving blob, as a share of the frame
MOTION_PAD = 0.25              # margin added around the moving area
MOTION_MIN_CROP = 0.3          # smallest crop sent to Mirage, as a share of the frame side
MOTION_FULL_FRAME = 0.6        # send the whole frame when motion covers more than this
MOTION_LEARN_SECONDS = 10.0    # how fast the background absorbs changes
MOTION_RESET_FRACTION = 0.7    # this much of the frame changing at once resets the background
MOTION_EXPOSURE = True         # cancel whole-frame brightness changes (auto exposure, clouds)
# Moving areas where Mirage then finds no vehicle (a "ghost" left by a car that
# was there when the background was learnt, a door, a shadow) fade into the
# background this much per frame instead of slowly. Areas touching any vehicle
# seen in the last MAX_UNSEEN_SECONDS are spared. 0 turns it off.
MOTION_FORGET_EMPTY = 0.3

# Parked vehicles: a vehicle that has not moved for PARKED_SECONDS is parked.
# It gets no second plate read, is absorbed into the motion background, and
# is not logged unless it moved at some point (or LOG_PARKED is on).
PARKED_SECONDS = 20.0
PARKED_MOVE = 0.03             # movement below this share of the frame does not count
LOG_PARKED = False

MAX_FPS = 20.0                 # cap on video frames per second (Mirage runs as fast as it can)
SHOW_WINDOW = True             # live preview window (press Q or Esc to stop)
OUTPUT_DIR = "traffic_output"
SAVE_SNAPSHOTS = True

# ----------------------------------------------------------- END SETTINGS --

# Prefer TCP for RTSP: UDP drops packets on busy networks and smears frames.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402  (must come after the environment variable above)
import numpy as np  # noqa: E402


# ------------------------------------------------------------ video input --

class SourceError(RuntimeError):
    """The camera or file could not be opened."""


class LatestFrameReader:
    """Reads a stream on a thread and always hands out the newest frame."""

    def __init__(self, source: str, log=None):
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
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
            raise SourceError(f"could not open video source: {redact(str(self.source))}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if not self.is_live:
                    self._ended.set()
                    return
                self.log("stream dropped, reconnecting in 2 s ...")
                self._cap.release()
                if self._stop.wait(2):
                    return
                try:
                    self._cap = self._open()
                except SourceError as exc:
                    self.log(str(exc))
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


def _words(label: str):
    return set(re.findall(r"[a-z]+", label.lower()))


def _has_word(label: str, words) -> bool:
    found = _words(label)
    return any(w in found or w + "s" in found or w + "es" in found for w in words)


def is_vehicle(label: str) -> bool:
    return _has_word(label, VEHICLE_WORDS) and not _has_word(label, VEHICLE_PART_WORDS)


def is_plate(label: str) -> bool:
    return any(phrase in label.lower() for phrase in PLATE_WORDS)


def is_ignored_region(label: str) -> bool:
    return _has_word(label, IGNORE_TEXT_IN)


def area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def contains_point(box, x, y) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def drop_duplicates(detections):
    """Remove vehicle boxes that are a second detection of a bigger box."""
    kept = []
    for det in sorted(detections, key=lambda d: area(d[2]), reverse=True):
        box = det[2]
        duplicate = False
        for other in kept:
            ob = other[2]
            inter = area((max(box[0], ob[0]), max(box[1], ob[1]), min(box[2], ob[2]), min(box[3], ob[3])))
            if (area(box) > 0 and inter / area(box) >= DUPLICATE_OVERLAP
                    and area(box) / area(ob) >= DUPLICATE_SIZE_RATIO):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


# ------------------------------------------------------------------ plates --

# Characters OCR commonly swaps; tried when a read does not fit the format.
_AMBIGUOUS = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S",
              "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G"}
# Position-aware fixes used with PLATE_FORMATS.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1",
             "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}


def clean_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9ÄÖÜ]", "", text.upper())


_LOOKALIKE = str.maketrans("OQDILZSBGT", "0001125861")


def is_ignored_text(cleaned: str) -> bool:
    """True if cleaned is, or looks like, an IGNORE_TEXT entry (CAMERA01, CAMERAO1, CAMERA0I...)."""
    shape = cleaned.translate(_LOOKALIKE)
    for entry in IGNORE_TEXT:
        target = clean_plate(entry).translate(_LOOKALIKE)
        if target and (shape == target or (len(target) >= 5 and (target in shape or
                                                                  (len(shape) >= 5 and shape in target)))):
            return True
    return False


def fit_format(cleaned: str, layout: str):
    """Fit cleaned to a layout such as AA99AAA. Returns (plate, fixes) or None."""
    if len(cleaned) != len(layout):
        return None
    out, fixes = [], 0
    for ch, want in zip(cleaned, layout.upper()):
        if want == "A":
            if ch.isalpha():
                out.append(ch)
            elif ch in _TO_LETTER:
                out.append(_TO_LETTER[ch])
                fixes += 1
            else:
                return None
        elif want == "9":
            if ch.isdigit():
                out.append(ch)
            elif ch in _TO_DIGIT:
                out.append(_TO_DIGIT[ch])
                fixes += 1
            else:
                return None
        elif ch == want:
            out.append(ch)
        else:
            return None
    return "".join(out), fixes


def _fit_formats(cleaned: str):
    best = None
    for layout in PLATE_FORMATS:
        # Also try every window, so a stray border mark or a country band
        # ("GB", "TR") before or after the plate does not spoil the read.
        for start in range(0, len(cleaned) - len(layout) + 1):
            fitted = fit_format(cleaned[start:start + len(layout)], layout)
            if fitted is None or fitted[1] > PLATE_MAX_FIXES:
                continue
            extra = len(cleaned) - len(layout)
            score = (fitted[1] + extra, -len(layout))
            if best is None or score < best[0]:
                best = (score, fitted[0])
    return [best[1]] if best else []


def plate_candidates(text: str):
    """Plates that text may contain, best guess first."""
    cleaned = clean_plate(text)
    if not cleaned or is_ignored_text(cleaned):
        return []
    if PLATE_FORMATS:
        return _fit_formats(cleaned)
    if re.match(PLATE_REGEX, cleaned):
        return [cleaned]
    # Try swapping look-alike characters (fewest swaps first, at most
    # PLATE_MAX_FIXES), but only for text that already mixes letters and
    # digits, so signs such as "STOP" stay out.
    if not (re.search(r"[0-9]", cleaned) and re.search(r"[A-ZÄÖÜ]", cleaned)):
        return []
    spots = [i for i, ch in enumerate(cleaned) if ch in _AMBIGUOUS]
    for count in range(1, min(PLATE_MAX_FIXES, len(spots)) + 1):
        for chosen in itertools.combinations(spots, count):
            fixed = list(cleaned)
            for i in chosen:
                fixed[i] = _AMBIGUOUS[fixed[i]]
            if re.match(PLATE_REGEX, "".join(fixed)):
                return ["".join(fixed)]
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
    plate_box: object = None   # plate Mirage found on this vehicle in the current frame
    anchor: object = None      # where it was when it last moved (normalised centre)
    last_moved: float = 0.0
    ever_moved: bool = False
    last_crop: float = 0.0     # when it last had a second plate read

    def is_parked(self, now: float) -> bool:
        return now - self.last_moved >= PARKED_SECONDS

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

    def update(self, detections, width: int, height: int, now: float, region=None):
        """Match detections to tracks. Returns (matched_tracks, finished_tracks).

        region is the part of the frame Mirage analysed: None for the whole
        frame, a box for a crop, or False when Mirage was skipped. A track
        outside it was not looked for, so instead of counting missed frames it
        ends once unseen for MAX_UNSEEN_SECONDS.
        """
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
            if track.anchor is None:
                track.anchor, track.last_moved = (cx, cy), now
            elif math.dist((cx, cy), track.anchor) > PARKED_MOVE:
                track.anchor, track.last_moved, track.ever_moved = (cx, cy), now, True
        finished = []
        live = {t.id for t, _ in matched}
        for tid in list(self.tracks):
            if tid not in live:
                track = self.tracks[tid]
                looked = region is None or (region is not False and contains_point(region, *centre(track.box)))
                if looked:
                    track.missed += 1
                if track.missed > MAX_MISSED or (not looked and now - track.last_seen > MAX_UNSEEN_SECONDS):
                    finished.append(self.tracks.pop(tid))
        return [t for t, _ in matched], finished

    def flush(self):
        finished = list(self.tracks.values())
        self.tracks.clear()
        return finished


# ------------------------------------------------------------ motion gate --

@dataclass
class Motion:
    """What the motion gate decided for one frame."""
    action: str                # "full" (whole frame), "crop" (region only) or "skip"
    reason: str
    region: object = None      # (x0, y0, x1, y1) in frame pixels for "crop"
    mask: object = None        # small uint8 mask of moving pixels, for display
    boxes: list = field(default_factory=list)


class MotionGate:
    """Learns a background image and finds where the frame differs from it.

    Works on a small blurred greyscale copy (WORK_WIDTH wide), so a frame costs
    about a millisecond. The background follows slow changes (daylight,
    parked cars) at a rate set by MOTION_LEARN_SECONDS, except inside boxes
    passed as protect, so a vehicle waiting at a barrier is not absorbed.
    """

    WORK_WIDTH = 320

    def __init__(self):
        self.background = None
        self.last_time = None
        self.grey = None           # the last frame, small and grey
        self.scale = 1.0
        self.ignore = []           # frame boxes whose changes never count (camera clock)

    def reset(self):
        self.background = None

    def absorb(self, boxes, rate: float):
        """Blend the background towards the last frame inside boxes (frame pixels)."""
        if self.background is None or self.grey is None or rate <= 0:
            return
        for box in boxes:
            x0, y0, x1, y1 = (int(v * self.scale) for v in box)
            x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
            bg, now = self.background[y0:y1 + 3, x0:x1 + 3], self.grey[y0:y1 + 3, x0:x1 + 3]
            bg += min(1.0, rate) * (now - bg)

    def update(self, frame, now: float, protect=()):
        h, w = frame.shape[:2]
        scale = min(1.0, self.WORK_WIDTH / w)
        small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        self.grey, self.scale = grey, scale
        if self.background is None or self.background.shape != grey.shape:
            self.background, self.last_time = grey, now
            return Motion("full", "learning the background")

        background = self.background
        if MOTION_EXPOSURE:
            # Auto exposure and clouds scale the whole picture; undo that before comparing.
            # The median ratio ignores the few pixels that really changed.
            gain = float(np.median((grey[::4, ::4] + 1.0) / (background[::4, ::4] + 1.0)))
            background = background * min(2.0, max(0.5, gain))
        mask = (cv2.absdiff(grey, background) > MOTION_THRESHOLD).astype(np.uint8) * 255
        if cv2.countNonZero(mask) > MOTION_RESET_FRACTION * mask.size:
            self.background, self.last_time = grey, now   # lights, day/night switch, camera moved
            return Motion("full", "scene changed, background reset", mask=mask)
        for box in self.ignore:
            x0, y0, x1, y1 = (int(v * scale) for v in box)
            mask[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        # Learn: the background drifts towards the frame, except under protected boxes.
        alpha = min(1.0, max(0.0, (now - self.last_time) / max(MOTION_LEARN_SECONDS, 1e-3)))
        self.last_time = now
        if alpha > 0:
            learn = np.full(grey.shape, 255, np.uint8)
            for box in protect:
                x0, y0, x1, y1 = (int(v * scale) for v in box)
                learn[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 0
            cv2.accumulateWeighted(grey, self.background, alpha, mask=learn)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw * bh >= MOTION_MIN_AREA * mask.size:
                boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))
        if not boxes:
            return Motion("skip", "no motion", mask=mask)

        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = MOTION_PAD * max(x1 - x0, y1 - y0)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        # Not too small, so Mirage still sees the whole vehicle around a moving part.
        min_w, min_h = MOTION_MIN_CROP * w, MOTION_MIN_CROP * h
        if x1 - x0 < min_w:
            cx = (x0 + x1) / 2
            x0, x1 = cx - min_w / 2, cx + min_w / 2
        if y1 - y0 < min_h:
            cy = (y0 + y1) / 2
            y0, y1 = cy - min_h / 2, cy + min_h / 2
        region = (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))
        if area(region) >= MOTION_FULL_FRAME * w * h:
            return Motion("full", "motion over most of the frame", mask=mask, boxes=boxes)
        return Motion("crop", "motion", region=region, mask=mask, boxes=boxes)


def cut_by_crop(box, region, width: int, height: int, margin: int = 3) -> bool:
    """True if box touches an edge of region that is not also the frame edge,
    i.e. Mirage saw only part of an object lying across the crop border."""
    x0, y0, x1, y1 = region
    return ((x0 > 0 and box[0] <= x0 + margin) or (y0 > 0 and box[1] <= y0 + margin)
            or (x1 < width and box[2] >= x1 - margin) or (y1 < height and box[3] >= y1 - margin))


def overlap(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def shift(box, dx: int, dy: int):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


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
            return None
        if not LOG_PARKED and not track.ever_moved:
            return None
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
        return row

    def close(self):
        self._file.close()


def crop_region(frame, box, pad: float):
    """Return (piece, (x0, y0)): box grown by pad, and its top-left corner."""
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = max(0, int(box[0] - bw * pad))
    y0 = max(0, int(box[1] - bh * pad))
    x1 = min(w, int(box[2] + bw * pad))
    y1 = min(h, int(box[3] + bh * pad))
    return frame[y0:y1, x0:x1], (x0, y0)


def crop(frame, box, pad: float):
    return crop_region(frame, box, pad)[0]


def draw(frame, tracks, fps: float, motion: Motion = None, now: float = None, mirage_fps: float = None):
    colours = {"coming": (0, 200, 0), "going": (0, 140, 255), "unknown": (200, 200, 200)}
    now = time.time() if now is None else now
    if SHOW_MOTION and motion is not None:
        if motion.mask is not None:
            big = cv2.resize(motion.mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            frame[big] = (0.55 * frame[big] + 0.45 * np.array((0, 0, 255))).astype(np.uint8)
        if motion.region is not None:
            r = motion.region
            cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 255), 2)
        cv2.putText(frame, f"Mirage: {motion.action} ({motion.reason})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for t in tracks:
        direction = "parked" if t.is_parked(now) else t.direction()
        colour = colours.get(direction, (130, 130, 130))
        x0, y0, x1, y1 = t.box
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        if t.plate_box is not None:
            p = t.plate_box
            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255, 255, 0), 1)
        plate = t.plate or "?"
        if t.plate and not t.plate_confirmed:
            plate += "?"
        text = f"#{t.id} {t.label} {direction} {plate}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, max(0, y0 - th - 8)), (x0 + tw + 6, y0), colour, -1)
        cv2.putText(frame, text, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    rates = f"{fps:.1f} fps" + (f"  Mirage {mirage_fps:.1f}/s" if mirage_fps is not None else "")
    cv2.putText(frame, f"{rates}  coming = {APPROACH}", (10, 25),
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


def owner_of(tracks, box):
    """The vehicle whose box holds box's centre; the smallest is the most specific."""
    cx, cy = centre(box)
    owners = [t for t in tracks if contains_point(t.box, cx, cy)]
    return min(owners, key=lambda t: area(t.box)) if owners else None


def split_objects(objects):
    """Split parsed objects into (vehicles, plate boxes, ignored-text boxes)."""
    vehicles = drop_duplicates([o for o in objects if is_vehicle(o[0])])
    plates = [o[2] for o in objects if is_plate(o[0])]
    ignored = [o[2] for o in objects if is_ignored_region(o[0])]
    return vehicles, plates, ignored


def assign_plates(tracks, plate_boxes):
    for track in tracks:
        track.plate_box = None
    for box in plate_boxes:
        owner = owner_of(tracks, box)
        if owner is not None and (owner.plate_box is None or area(box) > area(owner.plate_box)):
            owner.plate_box = box


def vote(track, text, box, ignored):
    """Add text read at box (frame pixels, or None) to track's plate votes."""
    if box is not None and any(contains_point(b, *centre(box)) for b in ignored):
        return
    weight = 1
    if box is not None and track.plate_box is not None and contains_point(track.plate_box, *centre(box)):
        weight = PLATE_BOX_VOTES
    for plate in plate_candidates(text):
        track.plate_votes[plate] += weight


def assign_text(tracks, lines, ignored=()):
    """Vote text lines whose centre falls inside a vehicle box for that vehicle."""
    for text, box in lines:
        if box is None:
            continue
        owner = owner_of(tracks, box)
        if owner is not None:
            vote(owner, text, box, ignored)


def crop_pass(vision, frame, track):
    """Read the plate (or the whole vehicle) again at full resolution.

    Returns [(text, box_in_frame_pixels_or_None), ...] for the caller to vote with.
    """
    if track.plate_box is not None and track.plate_box[2] - track.plate_box[0] >= PLATE_CROP_MIN_WIDTH:
        box, pad = track.plate_box, PLATE_CROP_PAD
    elif track.box[2] - track.box[0] >= CROP_MIN_WIDTH:
        box, pad = track.box, CROP_PAD
    else:
        return []
    piece, (ox, oy) = crop_region(frame, box, pad)
    if piece.size == 0:
        return []
    result = vision.describe(np.ascontiguousarray(piece[:, :, ::-1]), **mirage_options(["text"]))
    return [(text, shift(tbox, ox, oy) if tbox is not None else None)
            for text, tbox in parse_text(result, piece.shape[1], piece.shape[0])]


def parse_args(argv=None):
    global SOURCE, APPROACH, DUMP_JSON, BBOX_ORDER, SHOW_WINDOW, MAX_FPS, OUTPUT_DIR, PLATE_REGEX
    global PLATE_FORMATS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default=SOURCE, help="RTSP URL, video file or webcam number")
    p.add_argument("--approach", default=APPROACH, choices=["down", "up", "left", "right"],
                   help="screen direction a vehicle moves in when coming towards the camera")
    p.add_argument("--dump-json", action="store_true", default=DUMP_JSON,
                   help="print Mirage's raw JSON for the first frames")
    p.add_argument("--bbox-order", default=BBOX_ORDER, choices=["xyxy", "yxyx"])
    p.add_argument("--plate-regex", default=PLATE_REGEX)
    p.add_argument("--plate-format", action="append", default=None, metavar="LAYOUT",
                   help="plate layout such as AA99AAA (A = letter, 9 = digit); repeat for several")
    p.add_argument("--max-fps", type=float, default=MAX_FPS)
    p.add_argument("--no-window", action="store_true", help="run without the preview window")
    p.add_argument("--output", default=OUTPUT_DIR, help="folder for events.csv and snapshots")
    p.add_argument("--weights", default=None, help="path to mirage.cact (default: download)")
    args = p.parse_args(argv)
    SOURCE, APPROACH, DUMP_JSON = args.source, args.approach, args.dump_json
    BBOX_ORDER, MAX_FPS, OUTPUT_DIR = args.bbox_order, args.max_fps, args.output
    SHOW_WINDOW = SHOW_WINDOW and not args.no_window
    try:
        re.compile(args.plate_regex)
    except re.error as exc:
        p.error(f"--plate-regex is not a valid regular expression: {exc}")
    PLATE_REGEX = args.plate_regex
    if args.plate_format:
        bad = [f for f in args.plate_format if not re.fullmatch(r"[A9]+", f.upper())]
        if bad:
            p.error(f"plate layouts use only A and 9: {', '.join(bad)}")
        PLATE_FORMATS = tuple(f.upper() for f in args.plate_format)
    return args


def format_event(row) -> str:
    return (f"[event] #{row['track_id']} {row['vehicle']:<10} {row['direction']:<8} "
            f"plate={row['plate'] or '-'} ({row['plate_votes']} votes)")


class Monitor:
    """The detect, track and read loop, shared by the command line and the GUI.

    Settings are read from this module's globals on every frame, so changing
    one (for example ``traffic_monitor.APPROACH = "up"``) applies while it runs.

    For a live stream, reading frames, the motion gate and on_frame run at the
    stream's rate (up to MAX_FPS) while Mirage runs on a worker thread, taking
    the newest frame whenever it is free. The video stays smooth however slow
    Mirage is. A video file is analysed frame by frame instead, so a recording
    gives the same result every time.

    on_frame(frame_bgr, visible_tracks, stats, motion) is called after every
    frame; return True from it to stop. on_event(row) is called for every row
    written to events.csv, and log(message) for status messages.
    """

    def __init__(self, vision, source, output_dir, on_frame=None, on_event=None, log=print):
        self.vision, self.source, self.output_dir = vision, source, output_dir
        self.on_frame, self.on_event, self.log = on_frame, on_event, log
        self.stats = {"processed": 0, "fps": 0.0, "vehicles": 0, "events": 0,
                      "mirage_frames": 0, "mirage_fps": 0.0}
        self.gate = MotionGate()
        self.tracker = Tracker()
        self.lock = threading.RLock()       # guards tracker, gate and the event log
        self._events = None
        self._job = None
        self._busy = False
        self._wake = threading.Condition()
        self._last_mirage = None

    # -- events ------------------------------------------------------------

    def _emit(self, finished):
        for track in finished:
            row = self._events.write(track)
            if row is not None:
                self.stats["events"] += 1
                self.log(format_event(row))
                if self.on_event:
                    self.on_event(row)

    # -- Mirage, on the worker thread --------------------------------------

    def _analyse(self, frame, motion, now):
        stats = self.stats
        height, width = frame.shape[:2]
        region = motion.region if motion.action == "crop" else None
        x0, y0 = (region[0], region[1]) if region else (0, 0)
        image = frame[region[1]:region[3], region[0]:region[2]] if region else frame
        rgb = np.ascontiguousarray(image[:, :, ::-1])  # OpenCV gives BGR
        result = self.vision.describe(rgb, **mirage_options(["objects", "text"]))
        stats["mirage_frames"] += 1
        if DUMP_JSON and stats["mirage_frames"] <= DUMP_JSON_FRAMES:
            self.log(json.dumps(result, ensure_ascii=False, indent=2)[:6000])
        ih, iw = image.shape[:2]
        objects = [(label, score, shift(box, x0, y0)) for label, score, box in parse_objects(result, iw, ih)]
        lines = [(text, shift(box, x0, y0) if box else None) for text, box in parse_text(result, iw, ih)]
        vehicles, plate_boxes, ignored = split_objects(objects)
        if region:   # half a parked car at the crop border is not a moving vehicle
            vehicles = [v for v in vehicles if not cut_by_crop(v[2], region, width, height)]

        with self.lock:
            if ignored:
                self.gate.ignore = ignored   # the camera's clock is not motion
            active, finished = self.tracker.update(vehicles, width, height, now, region=region)
            if motion.boxes:
                # Spare any vehicle seen recently, even if Mirage missed it this frame.
                recent = [t.box for t in self.tracker.tracks.values() if now - t.last_seen <= MAX_UNSEEN_SECONDS]
                empty = [b for b in motion.boxes if not any(area(overlap(b, r)) > 0 for r in recent)]
                self.gate.absorb(empty, MOTION_FORGET_EMPTY)
            assign_plates(active, plate_boxes)
            assign_text(active, lines, ignored)
            for track in active:
                size = area(track.box)
                if SAVE_SNAPSHOTS and size > track.best_area:
                    track.best_area, track.best_snapshot = size, crop(frame, track.box, CROP_PAD).copy()
            # One second read per Mirage pass at most, to the vehicle that waited longest.
            wanting = [t for t in active if not t.plate_confirmed and not t.is_parked(now)]
            chosen = None
            if CROP_PASS and wanting and stats["mirage_frames"] % max(1, CROP_EVERY) == 0:
                chosen = min(wanting, key=lambda t: t.last_crop)
                chosen.last_crop = now
            self._emit(finished)
        if chosen is not None:
            read = crop_pass(self.vision, frame, chosen)
            with self.lock:
                for text, box in read:
                    vote(chosen, text, box, ignored)

        done = time.monotonic()
        if self._last_mirage is not None:
            rate = 1.0 / max(done - self._last_mirage, 1e-6)
            stats["mirage_fps"] = rate if stats["mirage_frames"] <= 2 else 0.8 * stats["mirage_fps"] + 0.2 * rate
        self._last_mirage = done

    def _worker(self, stop, failures):
        while True:
            with self._wake:
                while self._job is None and not stop.is_set():
                    self._wake.wait(0.2)
                if self._job is None:
                    return
                job, self._job = self._job, None
            try:
                self._analyse(*job)
            except BaseException as exc:  # report it on the main thread
                failures.append(exc)
                stop.set()
            finally:
                with self._wake:
                    self._busy = False
                    self._wake.notify_all()

    def _submit(self, frame, motion, now):
        with self._wake:
            self._busy = True
            self._job = (frame, motion, now)
            self._wake.notify_all()

    def _wait_idle(self, stop):
        with self._wake:
            while self._busy and not stop.is_set():
                self._wake.wait(0.2)

    # -- main loop ---------------------------------------------------------

    def run(self, stop: threading.Event = None):
        stop = stop or threading.Event()
        self.log(f"opening {redact(str(self.source))}")
        reader = LatestFrameReader(str(self.source), log=self.log)
        self._events = EventLog(self.output_dir)
        self.log(f"logging to {self._events.path.resolve()}")
        stats, tracker = self.stats, self.tracker
        seq, fps, last_done = 0, 0.0, None
        worker_stop, failures = threading.Event(), []
        worker = threading.Thread(target=self._worker, args=(worker_stop, failures), daemon=True)
        worker.start()
        try:
            while not stop.is_set() and not failures:
                started = time.monotonic()
                seq, frame = reader.read(seq, timeout=0.5)
                if frame is None:
                    if reader.is_live:
                        continue  # waiting or reconnecting
                    self.log("end of video")
                    break
                height, width = frame.shape[:2]
                now = time.time()
                with self.lock:
                    if MOTION_GATE:
                        protect = [t.box for t in tracker.tracks.values()
                                   if now - t.last_seen < MAX_UNSEEN_SECONDS and not t.is_parked(now)]
                        motion = self.gate.update(frame, now, protect)
                    else:
                        self.gate.reset()   # start afresh if the gate is switched back on
                        motion = Motion("full", "motion gate off")

                with self._wake:
                    idle = not self._busy
                if idle:
                    if motion.action == "skip":
                        with self.lock:
                            _, finished = tracker.update([], width, height, now, region=False)
                            self._emit(finished)
                    else:
                        self._submit(frame, motion, now)
                        if not reader.is_live:
                            self._wait_idle(stop)   # a recording: analyse every frame in turn
                elif motion.action != "skip":
                    motion = Motion("busy", "Mirage busy with an earlier frame", mask=motion.mask,
                                    boxes=motion.boxes)

                stats["processed"] += 1
                done = time.monotonic()
                elapsed = done - started
                # Real throughput: time between finished frames, waits included.
                rate = 1.0 / max(done - (last_done or started), 1e-6)
                last_done = done
                fps = rate if stats["processed"] == 1 else 0.9 * fps + 0.1 * rate
                with self.lock:
                    stats["fps"], stats["vehicles"] = fps, len(tracker.tracks)
                    visible = [snapshot(t) for t in tracker.tracks.values() if now - t.last_seen <= 1.0]
                if self.on_frame and self.on_frame(frame, visible, dict(stats), motion):
                    break
                min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
                if min_interval > elapsed:
                    stop.wait(min_interval - elapsed)
            if failures:
                raise failures[0]
        finally:
            worker_stop.set()
            with self._wake:
                self._wake.notify_all()
            worker.join(timeout=30)
            with self.lock:
                self._emit(tracker.flush())
                self._events.close()
            reader.close()


def snapshot(track):
    """A copy of a track that is safe to draw while the worker updates the original."""
    ends = track.history[:1] + track.history[-1:] if len(track.history) > 1 else list(track.history)
    return replace(track, history=ends, plate_votes=collections.Counter(track.plate_votes))


class MissingPackage(RuntimeError):
    """A package this program needs is not installed in the running Python."""


_PIP_NAMES = {"cactus": "cactus-compute", "PIL": "Pillow", "cv2": "opencv-python",
              "huggingface_hub": "huggingface_hub", "numpy": "numpy"}


def missing_message(module: str) -> str:
    package = _PIP_NAMES.get((module or "").split(".")[0], module)
    return (f"Python can't find the package '{package}'.\n\n"
            f"The Python running this program is:\n{sys.executable}\n\n"
            "In Visual Studio, open Solution Explorer > Python Environments. The environment "
            "you created (env) must be the active one, shown in bold: right-click it > "
            "Activate Environment. If it is missing, right-click Python Environments > "
            "Add Environment... > Existing environment and pick the env folder. Then "
            "right-click the environment > Install from requirements.txt, and run again.")


def load_mirage():
    """Return the cactus.Mirage class.

    Uses the installed cactus-compute package, or else the cactus clone this
    example lives in (two folders up), so it still runs when the package is
    missing from the active Python environment.
    """
    try:
        from cactus import Mirage
        return Mirage
    except ModuleNotFoundError as exc:
        if exc.name != "cactus":
            raise MissingPackage(missing_message(exc.name)) from exc
    repo = Path(__file__).resolve().parents[2]
    if (repo / "cactus" / "__init__.py").is_file():
        if str(repo) not in sys.path:
            sys.path.append(str(repo))
        try:
            from cactus import Mirage
            return Mirage
        except ModuleNotFoundError as exc:
            raise MissingPackage(missing_message(exc.name or "cactus")) from exc
    raise MissingPackage(missing_message("cactus"))


def create_mirage(weights=None):
    """Load Mirage, turning a missing package into a MissingPackage error."""
    Mirage = load_mirage()
    try:
        return Mirage(weights=weights)
    except ModuleNotFoundError as exc:  # e.g. huggingface_hub, needed to download
        raise MissingPackage(missing_message(exc.name)) from exc


def missing_packages():
    """Packages the first Start will need but the running Python lacks."""
    import importlib.util
    missing = []
    try:
        load_mirage()
    except MissingPackage:
        missing.append("cactus-compute")
    if importlib.util.find_spec("huggingface_hub") is None:
        missing.append("huggingface_hub")
    return missing


def main(argv=None):
    args = parse_args(argv)
    print("loading Mirage (the first run downloads the engine and weights) ...")
    try:
        vision = create_mirage(args.weights)
    except MissingPackage as exc:
        raise SystemExit(str(exc))
    window = {"ok": SHOW_WINDOW}

    def show(frame, active, stats, motion):
        if window["ok"]:
            try:
                cv2.imshow("Traffic monitor (Q to quit)",
                           draw(frame.copy(), active, stats["fps"], motion, mirage_fps=stats["mirage_fps"]))
                return cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27)
            except cv2.error:
                print("no display available; continuing without the preview window")
                window["ok"] = False
        elif stats["processed"] % 50 == 0:
            share = stats["mirage_frames"] / max(1, stats["processed"])
            print(f"{stats['processed']} frames, {stats['fps']:.1f} fps, {stats['vehicles']} vehicles in view, "
                  f"Mirage ran on {share:.0%} of frames")
        return False

    try:
        Monitor(vision, SOURCE, OUTPUT_DIR, on_frame=show).run()
    except SourceError as exc:
        raise SystemExit(str(exc))
    except KeyboardInterrupt:
        print("stopping")
    finally:
        if window["ok"]:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
