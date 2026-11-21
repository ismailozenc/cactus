# Traffic monitor (RTSP + Mirage)

Reads an RTSP camera, detects vehicles with Mirage, reads their number plates
and decides whether each vehicle is **coming** towards the camera or **going**
away. Every vehicle that leaves the picture becomes one row in
`traffic_output/events.csv`, with a snapshot in `traffic_output/snapshots/`.

| File | What it is |
| --- | --- |
| `traffic_monitor.py` | The program. Settings are at the top of the file. |
| `test_traffic_monitor.py` | Offline tests with a fake Mirage and a synthetic video. |
| `TrafficMonitor.sln`, `TrafficMonitor.pyproj` | Visual Studio solution and Python project. |
| `requirements.txt` | `cactus-compute`, `opencv-python`, `numpy`. |

## Open and run it in Visual Studio Community (no command line)

1. **Add Python support** (once). Open **Visual Studio Installer** from the
   Start menu, click **Modify** on Visual Studio Community, tick the
   **Python development** workload and click **Modify**. If you have no
   Python 3.9 or newer installed, also tick a Python 3 interpreter under
   **Individual components**.
2. **Open the project.** In Visual Studio choose **File > Open >
   Project/Solution...** and pick `examples\traffic_monitor\TrafficMonitor.sln`
   inside your clone.
3. **Create a virtual environment with the packages.** In **Solution
   Explorer**, right-click **Python Environments** > **Add Environment...**.
   Choose **Virtual environment**, leave the location as `env`, make sure
   **Install packages from file** points to `requirements.txt`, and click
   **Create**. Visual Studio installs everything and makes it the project's
   environment.
4. **Point it at your camera.** Open `traffic_monitor.py` and edit the
   settings block at the top. The main ones are listed under
   [Settings](#settings) below.
5. **Run it.** Press **F5** (**Debug > Start Debugging**) or **Ctrl+F5**
   (**Debug > Start Without Debugging**, which runs faster). A console window
   shows the log and a preview window shows the boxes. Press **Q** or **Esc**
   in the preview window to stop.

The first run downloads the Cactus engine (a Windows x86-64 or ARM64 build
is published) and the Mirage weights from Hugging Face, then caches them.

**Arguments without the command line:** **Project > TrafficMonitor
Properties > Debug > Script Arguments** takes the same options as the
command line, for example
`--source rtsp://user:pass@192.168.1.64/stream --approach up --dump-json`.

**Running the tests from menus:** **Test > Test Explorer**. If Visual
Studio asks, pick **pytest** as the framework and the project folder as the
root, then click **Run All**.

**Using your clone's Cactus code instead of PyPI:** right-click **Search
Paths** in Solution Explorer > **Add Folder to Search Path...** and choose the
root folder of the clone. Also install the clone's `requirements.txt`
(`huggingface_hub`, `Pillow`) into the environment.

## Settings

| Setting | Meaning |
| --- | --- |
| `SOURCE` | RTSP URL, a video file, or a webcam number such as `"0"`. A recording is the easiest way to tune. |
| `APPROACH` | Screen direction a vehicle moves in when it is coming towards you: `down`, `up`, `left`, `right`. |
| `PLATE_REGEX` | Your country's plate format after removing spaces and dashes. Tightening it removes most misreads. |
| `PLATE_CONFIRM_VOTES` | Identical reads needed before a plate counts as confirmed. |
| `DUMP_JSON` | Prints Mirage's raw JSON for the first frames. Use it on your first run. |
| `BBOX_ORDER` | `xyxy` or `yxyx`. Switch it if boxes are drawn in the wrong place. |
| `CROP_PASS`, `CROP_EVERY`, `CROP_MIN_WIDTH` | Second, full-resolution read of each unconfirmed vehicle. Costs speed. |
| `MAX_FPS`, `SHOW_WINDOW`, `OUTPUT_DIR` | Frame cap, preview window, and output folder. |

## First run checklist

1. **Turn on `DUMP_JSON`.** Mirage's exact output key names are not
   documented in this repository. The parser accepts the common ones
   (`label`/`name`/`class`, `bbox`/`box`, `text`), but check the dump. If
   vehicles are never detected, the label or box key probably differs.
   Adjust `LABEL_KEYS`, `BOX_KEYS` or `TEXT_KEYS` near the top of the parsing
   code, or the vehicle words in `VEHICLE_WORDS`.
2. **Check the boxes line up** with the vehicles in the preview. If they do
   not, set `BBOX_ORDER = "yxyx"`.
3. **Set `PLATE_REGEX`** for your country.
4. **Set `APPROACH`** by watching one vehicle come towards the camera.

## How direction and plates work

* **Direction** uses the whole life of a track. Moving in the `APPROACH`
  direction and getting bigger means *coming*. The reverse means *going*.
  Too little movement gives *unknown*. One frame can't tell motion, so
  this needs at least two frames of the same vehicle.
* **Plates** come from text Mirage reads inside a vehicle's box, plus a
  second read of a full-resolution crop, because Mirage shrinks its input
  and plates are small. Each read is cleaned, checked against
  `PLATE_REGEX`, gets one common OCR fix tried (O/0, I/1, S/5, B/8, Z/2,
  G/6), and votes. The most-voted plate wins.

Mirage is a general vision model, not a dedicated plate reader. Test it on
recordings from your own camera before relying on it, especially at night,
at a steep angle or at a distance.
