"""Traffic monitor window: live video, settings you can change while it runs,
and a list of every vehicle that passed.

Run this file (F5 in Visual Studio). Type the camera address at the top and
press Start. Every setting on the right applies immediately, except the camera,
output folder and weights, which apply the next time you press Start. Settings
are saved to traffic_settings.json when you press Start and when you close
the window.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import traffic_monitor as tm  # noqa: E402

import cv2  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

SETTINGS_FILE = HERE / "traffic_settings.json"

# Country: (plate layouts, plate regex). Layouts win when set; the regex is
# used when there are none.
PLATE_PRESETS = {
    "Germany": ((), tm.GERMAN_PLATE_REGEX),
    "United Kingdom": (("AA99AAA",), tm.ANY_PLATE_REGEX),
    "Turkey": (("99A9999", "99A99999", "99AA999", "99AA9999", "99AAA99", "99AAA999"), tm.ANY_PLATE_REGEX),
    "Any (letters and digits)": ((), tm.ANY_PLATE_REGEX),
}
CUSTOM = "Custom"

# Saved values that were defaults in earlier versions; a saved file holding
# one of these gets the current default instead, so new defaults reach
# people who never changed the setting.
OLD_DEFAULTS = {
    "PLATE_REGEX": [r"^(?=.*[0-9])(?=.*[A-Z])[A-Z0-9]{4,9}$"],
    "MAX_FPS": [10.0],
    "CROP_EVERY": [2],
    "MIN_TRACK_FRAMES": [3],
}

# (setting, kind, label, help, extra). Kinds: choice, int, float, optfloat
# (blank = model default), bool, text, regex, list (comma separated), layouts.
SETTINGS = {
    "Direction": [
        ("APPROACH", "choice", "Coming = moving", "Which way a vehicle moves on screen when it drives towards the camera.",
         ("down", "up", "left", "right")),
        ("DIRECTION_THRESHOLD", "float", "Movement needed", "How far a vehicle must move (share of the frame) before it is called "
         "coming or going. Raise it if parked cars get a direction, lower it if moving cars stay 'unknown'.", (0.0, 1.0, 0.01)),
        ("IOU_MATCH", "float", "Box overlap to follow", "How much a box must overlap the last one to count as the same "
         "vehicle. Lower it for fast traffic or a low frame rate.", (0.05, 0.95, 0.05)),
        ("MATCH_DISTANCE", "float", "Max move between passes", "How far (in box sizes) a vehicle may move between "
         "two Mirage passes and still count as the same vehicle, when the boxes no longer overlap. Raise it if "
         "Mirage is slow and cars get new numbers as they drive.", (0.0, 10.0, 0.25)),
        ("MAX_MISSED", "int", "Frames before a car is gone", "Frames a vehicle may go undetected before it is logged and "
         "forgotten.", (1, 500)),
        ("MIN_TRACK_FRAMES", "int", "Shortest track to log", "Vehicles seen in fewer Mirage passes than this are "
         "treated as noise. Keep it low if Mirage is slow (see the seconds per pass in the status bar).",
         (1, 500)),
        ("MAX_UNSEEN_SECONDS", "float", "Forget after (s) unseen", "With the motion gate, a vehicle outside the area "
         "Mirage reads is logged and forgotten after this long.", (0.2, 600, 0.5)),
    ],
    "Plates": [
        ("_PRESET", "preset", "Country", "Sets the plate format for a country. Germany uses the regex "
         "(district, letters, digits, E/H for electric or historic).", None),
        ("PLATE_FORMATS", "layouts", "Plate layouts", "A = letter, 9 = digit, comma separated, e.g. AA99AAA. Reads are "
         "corrected to fit (BSLIZ6B becomes BS11ZGB). Leave empty to use the regex below.", None),
        ("PLATE_MAX_FIXES", "int", "Max characters corrected", "How many characters a layout may correct in one read.",
         (0, 10)),
        ("PLATE_REGEX", "regex", "Plate regex", "Used only when no layouts are set: a regular expression the cleaned "
         "plate must match.", None),
        ("PLATE_CONFIRM_VOTES", "int", "Votes to confirm", "Identical reads needed before a plate counts as confirmed. "
         "Higher means fewer wrong plates but more misses.", (1, 100)),
        ("PLATE_BOX_VOTES", "int", "Votes for plate-box text", "Votes a read inside Mirage's 'license plate' box is worth.",
         (1, 20)),
        ("IGNORE_TEXT", "list", "Never a plate", "Text that is never a plate, such as your camera's on-screen name. "
         "Comma separated.", None),
    ],
    "Reading": [
        ("CROP_PASS", "bool", "Second read at full resolution", "Read each unconfirmed plate again from a crop of the "
         "full-resolution frame. Much better for small plates, but slower.", None),
        ("CROP_EVERY", "int", "Second read every N passes", "Each Mirage pass gives at most one vehicle a second "
         "read; this does it only every Nth pass. Raise it if Mirage is too slow.", (1, 100)),
        ("PLATE_CROP_PAD", "float", "Plate crop margin", "Extra margin around the plate box for the second read.",
         (0.0, 2.0, 0.05)),
        ("PLATE_CROP_MIN_WIDTH", "int", "Smallest plate to crop (px)", "Plate boxes narrower than this use the vehicle "
         "crop instead.", (0, 4000)),
        ("CROP_MIN_WIDTH", "int", "Smallest vehicle to crop (px)", "Vehicles narrower than this are too far away for "
         "a second read.", (0, 8000)),
        ("CROP_PAD", "float", "Vehicle crop margin", "Extra margin around the vehicle box.", (0.0, 1.0, 0.01)),
        ("TEXT_CONF", "optfloat", "Text confidence", "Minimum confidence for text. Leave blank for Mirage's default.",
         (0.0, 1.0, 0.05)),
    ],
    "Detection": [
        ("VEHICLE_WORDS", "list", "Vehicle names", "Mirage object names that count as vehicles. Comma separated.", None),
        ("VEHICLE_PART_WORDS", "list", "Vehicle parts", "Names containing one of these words are parts, not vehicles.",
         None),
        ("PLATE_WORDS", "list", "Plate names", "Object names Mirage uses for number plates.", None),
        ("IGNORE_TEXT_IN", "list", "Ignore text inside", "Text inside objects with these names is skipped (Mirage calls "
         "the camera's on-screen text 'watermark').", None),
        ("OBJECT_CONF", "optfloat", "Object confidence", "Minimum confidence for objects. Leave blank for Mirage's "
         "default. Raise it to cut false detections.", (0.0, 1.0, 0.05)),
        ("DUPLICATE_OVERLAP", "float", "Duplicate overlap", "A vehicle box this much inside a bigger one is a duplicate...",
         (0.0, 1.0, 0.05)),
        ("DUPLICATE_SIZE_RATIO", "float", "Duplicate size ratio", "...unless it is smaller than this share of the bigger "
         "box, which means a car further away.", (0.0, 1.0, 0.05)),
        ("BBOX_ORDER", "choice", "Box order", "Switch to yxyx only if boxes are drawn in the wrong place.",
         ("xyxy", "yxyx")),
    ],
    "Motion": [
        ("MOTION_GATE", "bool", "Only run Mirage where things move", "Learns what the empty scene looks like and "
         "runs Mirage only on the part of the frame that changed, or not at all when nothing moves. Saves most "
         "of the CPU and reads moving plates at higher resolution.", None),
        ("SHOW_MOTION", "bool", "Show motion on the video", "Tints moving pixels red and outlines in yellow the part "
         "Mirage reads. Use it while tuning the settings below.", None),
        ("MOTION_THRESHOLD", "int", "Change that counts (0-255)", "How much a pixel's brightness must differ from the "
         "background to count as motion. Raise it if rain, noise or swaying trees trigger Mirage.", (1, 255)),
        ("MOTION_MIN_AREA", "float", "Smallest moving blob", "Moving areas smaller than this share of the frame are "
         "ignored (insects, leaves, the camera clock).", (0.0, 0.5, 0.001)),
        ("MOTION_PAD", "float", "Margin around motion", "Extra margin around the moving area, so Mirage sees the "
         "whole vehicle.", (0.0, 2.0, 0.05)),
        ("MOTION_MIN_CROP", "float", "Smallest crop", "The crop sent to Mirage is at least this share of the frame's "
         "width and height.", (0.05, 1.0, 0.05)),
        ("MOTION_FULL_FRAME", "float", "Whole frame above", "When motion covers more than this share of the frame, "
         "the whole frame is sent.", (0.05, 1.0, 0.05)),
        ("MOTION_LEARN_SECONDS", "float", "Background adapts in (s)", "How quickly lasting changes (daylight, a "
         "newly parked car) become part of the background. Vehicles being tracked are never absorbed while they "
         "are not parked.", (0.5, 3600, 1)),
        ("MOTION_EXPOSURE", "bool", "Ignore brightness changes", "Cancels changes that brighten or darken the whole "
         "picture (the camera's auto exposure, clouds), so they are not taken for motion.", None),
        ("MOTION_RESET_FRACTION", "float", "Reset when this much changes", "If this share of the frame changes at "
         "once (lights, day/night switch), the background starts again.", (0.05, 1.0, 0.05)),
        ("MOTION_FORGET_EMPTY", "float", "Fade motion with no vehicle", "When Mirage finds no vehicle where "
         "something changed (a ghost of a car that has left, a shadow, a door), that area fades into the "
         "background this much per frame. 0 = off.", (0.0, 1.0, 0.1)),
        ("PARKED_SECONDS", "float", "Parked after (s)", "A vehicle that has not moved for this long is parked: no "
         "second plate read, and it fades into the background.", (1, 86400, 1)),
        ("PARKED_MOVE", "float", "Movement that counts", "Moves smaller than this share of the frame do not count "
         "(box jitter).", (0.0, 0.5, 0.005)),
        ("LOG_PARKED", "bool", "Log vehicles that never moved", "Off: parked cars are not written to events.csv.",
         None),
    ],
    "Output": [
        ("OUTPUT_DIR", "text", "Output folder *", "Folder for events.csv and snapshots. Applies at the next Start.",
         "folder"),
        ("SAVE_SNAPSHOTS", "bool", "Save a snapshot of each vehicle", "Saves one image per vehicle, taken when it was "
         "biggest.", None),
        ("MAX_FPS", "float", "Max video frames per second", "Cap on the video and motion check. Mirage runs on the "
         "newest frame whenever it is free, as fast as the computer allows. 0 = no cap.", (0, 60, 1)),
        ("DUMP_JSON", "bool", "Show Mirage's raw output in the log", "Prints the JSON Mirage returns for the first "
         "frames after Start.", None),
        ("DUMP_JSON_FRAMES", "int", "Frames to show", "How many frames of raw output to print.", (1, 1000)),
        ("WEIGHTS", "text", "Mirage weights file *", "Path to mirage.cact. Leave blank to download it. Applies at the "
         "next Start.", "file"),
    ],
}
START_ONLY = {"SOURCE", "OUTPUT_DIR", "WEIGHTS"}
tm.WEIGHTS = ""  # GUI-only setting


def to_text(kind, value) -> str:
    if kind in ("list", "layouts"):
        return ", ".join(value)
    if kind == "optfloat":
        return "" if value is None else f"{value:g}"
    if kind == "float":
        return f"{value:g}"
    return str(value)


def parse_value(kind, extra, text):
    """Turn a control's text into a setting value. Raises ValueError if invalid."""
    text = text.strip()
    if kind == "choice":
        if text not in extra:
            raise ValueError(f"choose one of {', '.join(extra)}")
        return text
    if kind == "int":
        value = int(text)
        if not extra[0] <= value <= extra[1]:
            raise ValueError(f"between {extra[0]} and {extra[1]}")
        return value
    if kind in ("float", "optfloat"):
        if kind == "optfloat" and not text:
            return None
        value = float(text)
        if not extra[0] <= value <= extra[1]:
            raise ValueError(f"between {extra[0]:g} and {extra[1]:g}")
        return value
    if kind == "regex":
        re.compile(text)
        return text
    if kind == "list":
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if kind == "layouts":
        layouts = tuple(part.strip().upper() for part in text.split(",") if part.strip())
        bad = [f for f in layouts if not re.fullmatch(r"[A9]+", f)]
        if bad:
            raise ValueError("layouts use only A (letter) and 9 (digit)")
        return layouts
    return text


def all_specs():
    for specs in SETTINGS.values():
        for spec in specs:
            if spec[1] != "preset":
                yield spec


DEFAULTS = {name: getattr(tm, name) for name, *_ in all_specs()}
DEFAULTS["SOURCE"] = tm.SOURCE


def load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    kinds = {name: (kind, extra) for name, kind, _, _, extra in all_specs()}
    kinds["SOURCE"] = ("text", None)
    for name, value in saved.items():
        if name not in kinds or value in OLD_DEFAULTS.get(name, []):
            continue
        kind, extra = kinds[name]
        if kind == "bool":
            if isinstance(value, bool):
                setattr(tm, name, value)
            continue
        try:
            text = to_text(kind, tuple(value) if isinstance(value, list) else value)
            setattr(tm, name, parse_value(kind, extra, text))
        except (TypeError, ValueError, re.error):
            pass  # ignore a stale or hand-edited value


def save_settings():
    data = {name: getattr(tm, name) for name in DEFAULTS}
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def open_path(path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606  (opens Explorer or the image viewer)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue = queue.Queue()
        self.vision, self.vision_weights = None, None
        self.worker, self.stop_event = None, threading.Event()
        self.monitor = None
        self.latest = None            # newest annotated frame from the worker
        self.latest_lock = threading.Lock()
        self.photo = None
        self.vars, self.errors = {}, {}
        self.output_dir = None
        self._syncing = False

        root.title("Traffic Monitor")
        root.geometry("1360x860")
        root.minsize(1000, 640)
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Help.TLabel", foreground="#555555")
        style.configure("Error.TLabel", foreground="#c62828")
        style.configure("Status.TLabel", padding=(8, 4))

        self._build_top()
        self._build_body()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self.poll)
        self.root.after(150, self._place_sashes)
        self.root.after(300, self._check_packages)
        self.root.after(33, self.refresh_video)
        self.set_running(False)

    # ------------------------------------------------------------ layout --

    def _build_top(self):
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Camera / video:").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value=tm.SOURCE)
        entry = ttk.Entry(bar, textvariable=self.source_var, width=60)
        entry.pack(side=tk.LEFT, padx=(6, 4), fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _e: self.start())
        ttk.Button(bar, text="Browse...", command=self.browse_video).pack(side=tk.LEFT, padx=(0, 10))
        self.start_btn = ttk.Button(bar, text="▶  Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(bar, text="■  Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(bar, text="Open output folder", command=self.open_output).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel",
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_body(self):
        vertical = self.vertical = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))
        top = self.top = ttk.PanedWindow(vertical, orient=tk.HORIZONTAL)
        vertical.add(top, weight=3)

        video_frame = ttk.Frame(top)
        self.video = tk.Label(video_frame, bg="#111111", fg="#bbbbbb",
                              text="Press Start to connect to the camera", font=("Segoe UI", 12))
        self.video.pack(fill=tk.BOTH, expand=True)
        top.add(video_frame, weight=3)

        side = ttk.Frame(top, padding=(8, 0, 0, 0))
        top.add(side, weight=1)
        # Bottom widgets first, so they stay visible and the tabs take the rest.
        self.help_var = tk.StringVar(value="Point at a setting to see what it does. "
                                           "Settings marked * apply at the next Start.")
        ttk.Button(side, text="Reset all settings to defaults", command=self.reset).pack(side=tk.BOTTOM, anchor=tk.E)
        ttk.Label(side, textvariable=self.help_var, style="Help.TLabel", wraplength=340,
                  justify=tk.LEFT).pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 4))
        notebook = ttk.Notebook(side)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for tab, specs in SETTINGS.items():
            page = ttk.Frame(notebook, padding=10)
            page.columnconfigure(1, weight=1)
            notebook.add(page, text=tab)
            for row, spec in enumerate(specs):
                self._add_control(page, row, *spec)
            if tab == "Motion":   # beside "Show motion on the video", where the tab has room
                row = [spec[0] for spec in specs].index("SHOW_MOTION")
                ttk.Button(page, text="Reset background", command=self.reset_background).grid(
                    row=row, column=1, sticky=tk.E)

        bottom = ttk.Notebook(vertical)
        vertical.add(bottom, weight=1)
        events = ttk.Frame(bottom)
        bottom.add(events, text="Vehicles")
        columns = ("time", "id", "vehicle", "direction", "plate", "votes", "confirmed", "snapshot")
        self.tree = ttk.Treeview(events, columns=columns, show="headings", height=6)
        widths = {"time": 150, "id": 50, "vehicle": 100, "direction": 90, "plate": 120, "votes": 60,
                  "confirmed": 95, "snapshot": 300}
        self.snapshots = {}
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=widths[col], anchor=tk.W, stretch=col == "snapshot")
        self.tree.tag_configure("coming", foreground="#1b7f2a")
        self.tree.tag_configure("going", foreground="#c45c00")
        scroll = ttk.Scrollbar(events, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<Double-1>", self.open_snapshot)

        logs = ttk.Frame(bottom)
        bottom.add(logs, text="Log")
        self.log_text = tk.Text(logs, height=6, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(logs, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_control(self, page, row, name, kind, label, help_text, extra):
        widgets = []
        if kind == "bool":
            var = tk.BooleanVar(value=bool(getattr(tm, name)))
            widget = ttk.Checkbutton(page, text=label, variable=var)
            widget.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=3)
        else:
            text_label = ttk.Label(page, text=label)
            text_label.grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
            widgets.append(text_label)
            if kind == "preset":
                var = tk.StringVar(value=self._preset_for(tm.PLATE_FORMATS, tm.PLATE_REGEX))
                widget = ttk.Combobox(page, textvariable=var, state="readonly",
                                      values=list(PLATE_PRESETS) + [CUSTOM], width=18)
            else:
                var = tk.StringVar(value=to_text(kind, getattr(tm, name)))
                if kind == "choice":
                    widget = ttk.Combobox(page, textvariable=var, values=list(extra), state="readonly", width=10)
                elif kind in ("int", "float", "optfloat"):
                    step = 1 if kind == "int" else extra[2]
                    widget = ttk.Spinbox(page, textvariable=var, from_=extra[0], to=extra[1],
                                         increment=step, width=10)
                else:
                    widget = ttk.Entry(page, textvariable=var, width=24)
            widget.grid(row=row, column=1, sticky=tk.EW, pady=3)
            if extra in ("folder", "file"):
                ttk.Button(page, text="...", width=3,
                           command=lambda n=name, e=extra: self.browse_setting(n, e)).grid(row=row, column=2)
            error = ttk.Label(page, text="", style="Error.TLabel")
            error.grid(row=row, column=3, sticky=tk.W, padx=(4, 0))
            self.errors[name] = error
        widgets.append(widget)
        self.vars[name] = (var, kind, extra)
        var.trace_add("write", lambda *_a, n=name: self.apply(n))
        for w in widgets:
            w.bind("<Enter>", lambda _e, t=help_text: self.help_var.set(t))

    def reset_background(self):
        if self.monitor is not None:
            self.monitor.gate.reset()
            self.log("Motion background reset; the next frame is learnt as the empty scene.")

    def _check_packages(self):
        self.log(f"Python: {sys.executable}")
        missing = tm.missing_packages()
        if not missing or (tm.WEIGHTS and missing == ["huggingface_hub"]):
            return  # huggingface_hub is only needed to download, not with a weights file
        self.status_var.set(f"Missing {' and '.join(missing)} - see the Log tab")
        message = tm.missing_message("cactus" if "cactus-compute" in missing else "huggingface_hub")
        self.log("WARNING: " + message)
        if messagebox.askyesno("Traffic Monitor", f"Missing: {', '.join(missing)}.\n\n"
                               f"Install everything in requirements.txt into this Python now?\n\n"
                               f"{sys.executable}\n\nChoose No to fix it yourself (instructions are "
                               f"in the Log tab)."):
            self.install_packages()

    def install_packages(self):
        """pip install -r requirements.txt into the running Python, output to the Log tab."""
        self.status_var.set("Installing packages (see the Log tab)...")
        self.start_btn.configure(state=tk.DISABLED)
        command = [sys.executable, "-m", "pip", "install", "-r", str(HERE / "requirements.txt")]

        def work():
            self.queue.put(("log", "> " + " ".join(command)))
            try:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, creationflags=flags)
                for line in proc.stdout:
                    self.queue.put(("log", line))
                code = proc.wait()
            except OSError as exc:
                self.queue.put(("log", str(exc)))
                code = -1
            self.queue.put(("installed", code))

        threading.Thread(target=work, daemon=True).start()

    def _place_sashes(self):
        self.root.update_idletasks()
        self.top.sashpos(0, max(400, self.top.winfo_width() - 470))
        self.vertical.sashpos(0, int(self.vertical.winfo_height() * 0.74))

    # ----------------------------------------------------------- settings --

    @staticmethod
    def _preset_for(layouts, regex):
        for preset, (values, pattern) in PLATE_PRESETS.items():
            if tuple(values) == tuple(layouts) and (values or pattern == regex):
                return preset
        return CUSTOM

    def _sync_preset(self):
        self._syncing = True
        self.vars["_PRESET"][0].set(self._preset_for(tm.PLATE_FORMATS, tm.PLATE_REGEX))
        self._syncing = False

    def apply(self, name):
        var, kind, extra = self.vars[name]
        if kind == "preset":
            if self._syncing or var.get() == CUSTOM:
                return
            layouts, pattern = PLATE_PRESETS[var.get()]
            self._syncing = True
            self.vars["PLATE_FORMATS"][0].set(", ".join(layouts))
            self.vars["PLATE_REGEX"][0].set(pattern)
            self._syncing = False
            return
        if kind == "bool":
            setattr(tm, name, bool(var.get()))
            return
        try:
            value = parse_value(kind, extra, var.get())
        except (ValueError, re.error) as exc:
            message = str(exc)
            if kind in ("int", "float", "optfloat") and "between" not in message:
                message = "enter a whole number" if kind == "int" else "enter a number"
            elif kind == "regex":
                message = "not a valid regular expression"
            self.errors[name].configure(text="⚠ " + message)
            return
        self.errors[name].configure(text="")
        setattr(tm, name, value)
        if name in ("PLATE_FORMATS", "PLATE_REGEX") and not self._syncing:
            self._sync_preset()

    def reset(self):
        if not messagebox.askyesno("Reset settings", "Put every setting back to its default?"):
            return
        for name, value in DEFAULTS.items():
            setattr(tm, name, value)
            if name in self.vars:
                var, kind, _ = self.vars[name]
                var.set(bool(value) if kind == "bool" else to_text(kind, value))
        self.source_var.set(DEFAULTS["SOURCE"])
        self._sync_preset()

    def browse_setting(self, name, what):
        if what == "folder":
            path = filedialog.askdirectory(initialdir=str(HERE))
        else:
            path = filedialog.askopenfilename(filetypes=[("Cactus archive", "*.cact"), ("All files", "*.*")])
        if path:
            self.vars[name][0].set(path)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if path:
            self.source_var.set(path)

    # ------------------------------------------------------------ running --

    def set_running(self, running: bool):
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning("Traffic Monitor", "Type a camera address (rtsp://...) or choose a video file.")
            return
        tm.SOURCE = source
        save_settings()
        output = Path(tm.OUTPUT_DIR)
        self.output_dir = output if output.is_absolute() else HERE / output
        self.stop_event = threading.Event()
        self.set_running(True)
        self.status_var.set("Starting...")
        self.worker = threading.Thread(target=self._work, args=(source, str(self.output_dir), tm.WEIGHTS or None),
                                       daemon=True)
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self.stop_btn.configure(state=tk.DISABLED)

    def _work(self, source, output, weights):
        post = lambda kind, payload=None: self.queue.put((kind, payload))  # noqa: E731
        try:
            if self.vision is None or self.vision_weights != weights:
                post("status", "Loading Mirage (the first run downloads the engine and weights)...")
                self.vision, self.vision_weights = tm.create_mirage(weights), weights
            post("status", "Connecting...")

            def on_frame(frame, visible, stats, motion):
                annotated = tm.draw(frame.copy(), visible, stats["fps"], motion, stats=stats)
                with self.latest_lock:
                    self.latest = annotated
                post("stats", stats)
                return False

            self.monitor = tm.Monitor(self.vision, source, output, on_frame=on_frame,
                                      on_event=lambda row: post("event", row),
                                      log=lambda message: post("log", message))
            self.monitor.run(self.stop_event)
        except tm.MissingPackage as exc:
            post("error", str(exc))
        except tm.SourceError as exc:
            post("error", f"{exc}\n\nCheck the address, that the camera or RTSP server is running, "
                          "and the user name and password.")
        except Exception as exc:  # show anything else instead of dying silently
            post("log", traceback.format_exc())
            post("error", f"{type(exc).__name__}: {exc}")
        finally:
            post("stopped")

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                    self.log(payload)
                elif kind == "stats":
                    share = payload["mirage_frames"] / max(1, payload["processed"])
                    self.status_var.set(f"Running  \u2022  video {payload['fps']:.1f} fps  \u2022  "
                                        f"{tm.mirage_status(payload)}, {payload['mirage_frames']} passes on "
                                        f"{share:.0%} of frames  \u2022  {payload['vehicles']} vehicles in view  "
                                        f"\u2022  {payload['events']} logged")
                elif kind == "event":
                    self.add_event(payload)
                elif kind == "error":
                    self.log("ERROR: " + payload)
                    messagebox.showerror("Traffic Monitor", payload)
                elif kind == "installed":
                    import importlib
                    importlib.invalidate_caches()
                    self.set_running(False)
                    missing = tm.missing_packages()
                    if payload == 0 and not missing:
                        self.status_var.set("Packages installed. Press Start.")
                        self.log("Packages installed.")
                    else:
                        self.status_var.set("Installing failed - see the Log tab")
                        messagebox.showerror("Traffic Monitor", "Installing the packages failed. The Log tab "
                                             "shows pip's output.\n\n" + tm.missing_message(
                                                 "cactus" if "cactus-compute" in missing else "huggingface_hub"))
                elif kind == "stopped":
                    self.set_running(False)
                    if not self.status_var.get().startswith("Stopped"):
                        self.status_var.set("Stopped")
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def refresh_video(self):
        with self.latest_lock:
            frame, self.latest = self.latest, None
        if frame is not None:
            width = max(self.video.winfo_width(), 50)
            height = max(self.video.winfo_height(), 50)
            scale = min(width / frame.shape[1], height / frame.shape[0])
            size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.configure(image=self.photo, text="")
        self.root.after(33, self.refresh_video)

    def add_event(self, row):
        snapshot = self.output_dir / row["snapshot"] if row["snapshot"] and self.output_dir else None
        iid = self.tree.insert("", 0, values=(row["first_seen"].replace("T", " "), row["track_id"], row["vehicle"],
                                              row["direction"], row["plate"] or "-", row["plate_votes"],
                                              "yes" if row["plate_confirmed"] else "no",
                                              snapshot.name + "  (double-click)" if snapshot else ""),
                               tags=(row["direction"],))
        self.snapshots[iid] = snapshot

    def open_snapshot(self, _event):
        selected = self.tree.selection()
        if selected:
            path = self.snapshots.get(selected[0])
            if path and path.exists():
                open_path(path)

    def open_output(self):
        folder = self.output_dir or (Path(tm.OUTPUT_DIR) if Path(tm.OUTPUT_DIR).is_absolute()
                                     else HERE / tm.OUTPUT_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        open_path(folder)

    def log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 5000:
            self.log_text.delete("1.0", f"{lines - 5000}.0")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def on_close(self):
        tm.SOURCE = self.source_var.get().strip() or tm.SOURCE
        save_settings()
        self.stop_event.set()
        self._close_when_stopped(50)

    def _close_when_stopped(self, tries):
        if self.worker and self.worker.is_alive() and tries > 0:
            self.root.after(100, self._close_when_stopped, tries - 1)
        else:
            self.root.destroy()


def main():
    if sys.platform == "win32":
        try:  # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    load_settings()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
