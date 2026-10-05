"""Background matting for the Studio Cam pipeline (omarchy-cam-studio).

Reads the real webcam at the stream size (--width/--height, 720p), scales each
frame down to the processing size (RES_FILE: 360/480/720, switchable live via
`studio-cam res` + SIGUSR1), runs Robust Video Matting and composites there in a
worker thread, then scales the result back up and writes it to a v4l2loopback
device that OBS uses as its source. The stream never changes size, so a switch
just looks softer or sharper. Output is synced to the mask: every frame sent is
the camera frame its alpha was computed from, repeated until the next one is
ready, so the mask never trails a moving person.

Output mode is read from MODE_FILE on SIGUSR1 (`studio-cam pause|camoff|live`):
  live    normal matted video
  paused  freezes on the last frame; the webcam keeps running so resume is instant
  loop    "motion pause": plays the last --loop-seconds of live video (at 1/--loop-hold speed),
          backward from now, then forward, over and over until resumed
  camoff  a "Camera off" card; the webcam is released (light goes off)
The virtual camera keeps streaming in every mode, so meeting apps never drop it.
If the webcam stops delivering frames, a "Camera unavailable" card is shown and
the camera is reopened until it comes back.
"""

import argparse
import glob
import os
import queue
import signal
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyvirtualcam

HERE = Path(__file__).resolve().parent
STATE_DIR = (Path(os.environ["XDG_RUNTIME_DIR"]) / "studio-cam" if os.environ.get("XDG_RUNTIME_DIR")
             else Path(f"/tmp/studio-cam-{os.getuid()}"))  # must match the studio-cam CLI
MODE_FILE = STATE_DIR / "mode"
RES_FILE = Path.home() / ".config" / "studio-cam" / "resolution"
CARD_FONT = "/usr/share/fonts/noto/NotoSans-Bold.ttf"  # falls back to PIL's default font if missing
MODES = ("live", "paused", "loop", "camoff")
# Processing sizes by height. Each needs models/rvm_resnet50_fp16_{w}x{h}_ds0.25.onnx
# (tools/build_models.py builds them).
PROC_SIZES = {360: (640, 360), 480: (854, 480), 720: (1280, 720)}
DEFAULT_PROC = 480
CAMERA_TIMEOUT = 2.0  # seconds without frames before showing "Camera unavailable"


def model_path(h):
    w, h = PROC_SIZES[h]
    return HERE / "models" / f"rvm_resnet50_fp16_{w}x{h}_ds0.25.onnx"


class Control:
    """What the CLI asked for, re-read from MODE_FILE and RES_FILE on SIGUSR1.

    Only plain attribute writes, so it is safe to update from the signal handler
    at any point, including before the worker threads exist.
    """
    mode = "live"
    res = DEFAULT_PROC
    limit = 720  # the stream height; processing is never finer than this

    @classmethod
    def reload(cls, *_):
        try:
            m = MODE_FILE.read_text().strip()
        except OSError:
            m = "live"
        try:
            r = int(RES_FILE.read_text().strip())
        except (OSError, ValueError):
            r = DEFAULT_PROC
        r = r if r in PROC_SIZES else DEFAULT_PROC
        cls.mode = m if m in MODES else "live"
        cls.res = max([h for h in PROC_SIZES if h <= min(r, cls.limit)] or [min(PROC_SIZES)])


def default_camera():
    """The first real webcam's capture node (loopback devices have no by-id entry)."""
    found = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    return found[0] if found else "/dev/video0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default=os.environ.get("STUDIO_CAM_CAMERA") or default_camera(),
                   help="webcam device (default: $STUDIO_CAM_CAMERA, else the first /dev/v4l/by-id/*-video-index0)")
    p.add_argument("--out", default="/dev/video11")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model", help="ONNX file stem in models/, overriding the processing size from RES_FILE")
    p.add_argument("--downsample", type=float, default=0.25, help="only used by dynamic-shape models")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--threads", type=int, default=6, help="CPU inference threads")
    p.add_argument("--blur", type=int, default=25, help="background blur radius in px at stream size; 0 = none")
    p.add_argument("--background", help="image file to use instead of blurring")
    p.add_argument("--loop-seconds", type=float, default=30, help="rolling buffer length for motion pause")
    p.add_argument("--loop-hold", type=int, default=1, help="output frames per buffered frame in motion pause; 2 = half speed")
    p.add_argument("--stats", action="store_true")
    a = p.parse_args()
    a.loop_hold = max(1, a.loop_hold)
    return a


class Camera(threading.Thread):
    """Reads the webcam; releases it in camoff and reopens it if it stops delivering frames."""

    def __init__(self, dev, w, h, fps):
        super().__init__(daemon=True)
        self.dev, self.w, self.h, self.fps = dev, w, h, fps
        self.cap = self.open()
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {dev}")
        self.cond = threading.Condition()
        self.frame, self.seq = None, 0
        self.opens = 1  # bumped on every (re)open, so Matter knows the scene may have changed
        self.last_time = time.monotonic()  # when the last frame arrived

    def open(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        fails, lost = 0, False
        while True:
            try:
                if Control.mode == "camoff":
                    self.release()
                    time.sleep(0.1)
                    continue
                if self.cap is None:
                    self.cap, self.opens = self.open(), self.opens + 1
                    if not self.cap.isOpened():
                        self.release()
                        time.sleep(1)
                        continue
                ok, f = self.cap.read()
                if not ok:
                    fails += 1
                    if fails >= 100:  # about a second of failed reads: unplugged, suspended or taken
                        if not lost:
                            print(f"camera {self.dev} stopped delivering frames; reopening", flush=True)
                        lost, fails = True, 0
                        self.release()
                        time.sleep(1)
                    else:
                        time.sleep(0.01)
                    continue
                if lost:
                    print("camera is back", flush=True)
                fails, lost = 0, False
                with self.cond:
                    self.frame, self.seq, self.last_time = f, self.seq + 1, time.monotonic()
                    self.cond.notify_all()
            except Exception as e:  # keep the thread alive; a dead reader would freeze the stream
                print(f"camera error: {e!r}", flush=True)
                self.release()
                time.sleep(1)

    def wait(self, after, timeout=0.2):
        """Next frame after `after`, or (None, after) if none arrives (e.g. camera released)."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > after, timeout):
                return None, after
            return self.frame, self.seq


def make_session(path, device, threads):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if device == "gpu":
        # GPU through the WebGPU plugin EP (Vulkan). NCHW layout avoids an ORT
        # layout-transform crash on RVM's graph.
        try:
            import onnxruntime_ep_webgpu as wg
        except ImportError:
            wg = None
            print("onnxruntime-ep-webgpu not installed, using the CPU", flush=True)
        if wg is not None:
            if not any(d.ep_name == wg.get_ep_name() for d in ort.get_ep_devices()):  # once per process
                ort.register_execution_provider_library(wg.get_ep_name(), wg.get_library_path())
            devs = [d for d in ort.get_ep_devices() if d.ep_name == wg.get_ep_name()]
            if devs:
                so.add_provider_for_devices(devs, {"preferredLayout": "NCHW"})
                return ort.InferenceSession(path, so)
            print("no WebGPU device found, using the CPU", flush=True)
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class Matter(threading.Thread):
    """Mattes and composites camera frames at the processing size.

    Each published result is one camera frame composited with the alpha
    computed from it, so the output stays in sync. Follows Control.res; a new
    session is built here and output holds its last frame meanwhile. Accepts
    the original dynamic-shape models (with a downsample_ratio input) or the
    fixed-shape fp16 ones from tools/build_models.py.
    """

    def __init__(self, cam, a):
        super().__init__(daemon=True)
        self.cam, self.a = cam, a
        self.loaded, self.failed = None, None  # processing heights: current, and last that failed to load
        self.bg_cache = {}  # background image per processing size
        self.result, self.seq = None, 0  # composited frame at stream size
        self.lock, self.rate = threading.Lock(), 0.0
        self.stale, self.cam_opens = False, cam.opens
        # Start at the saved size, or any size whose model exists.
        for h in [Control.res] + sorted(PROC_SIZES, reverse=True):
            if a.model or model_path(h).exists():
                self.load(h)
                break
        else:
            raise SystemExit("no matting model found in models/: run tools/build_models.py")

    def background(self, size):
        if self.a.bg_img is None:
            return None
        if size not in self.bg_cache:
            self.bg_cache[size] = cv2.resize(self.a.bg_img, size, interpolation=cv2.INTER_AREA)
        return self.bg_cache[size]

    def load(self, h):
        w, h = PROC_SIZES[h]
        path = HERE / "models" / f"{self.a.model}.onnx" if self.a.model else model_path(h)
        self.sess = make_session(str(path), self.a.device, self.a.threads)
        inputs = {i.name: i for i in self.sess.get_inputs()}
        shape = inputs["src"].shape
        self.size = (shape[3], shape[2]) if all(isinstance(d, int) for d in shape[2:]) else (w, h)
        self.dtype = np.float16 if "float16" in inputs["src"].type else np.float32
        self.zero_state = {n: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], self.dtype)
                           for n, i in inputs.items() if n.startswith("r")}
        self.feed = dict(self.zero_state)
        if "downsample_ratio" in inputs:
            self.feed["downsample_ratio"] = np.array([self.a.downsample], np.float32)
        self.loaded = h
        print(f"processing at {self.size[0]}x{self.size[1]} ({path.name})", flush=True)

    def run(self):
        seq, n, t0 = 0, 0, time.time()
        while True:
            try:
                want = Control.res
                if want not in (self.loaded, self.failed) and not self.a.model:
                    try:
                        self.load(want)
                    except Exception as e:  # e.g. missing model: keep matting at the current size
                        print(f"cannot switch to {want}p: {e}", flush=True)
                        self.failed = want
                if Control.mode != "live":
                    self.stale = True
                    time.sleep(0.1)
                    continue
                frame, seq = self.cam.wait(seq)
                if frame is None:
                    continue
                if self.stale or self.cam_opens != self.cam.opens:
                    # RVM's recurrent state remembers the scene; after a pause or a
                    # camera reopen that memory is wrong, so start fresh.
                    self.feed.update(self.zero_state)
                    self.stale, self.cam_opens = False, self.cam.opens
                small = frame if frame.shape[1::-1] == self.size else cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                # BGR uint8 HWC -> RGB float NCHW in [0, 1], in one pass
                self.feed["src"] = cv2.dnn.blobFromImage(small, 1 / 255.0, swapRB=True).astype(self.dtype)
                _, pha, *rec = self.sess.run(None, self.feed)
                for k, v in zip(("r1i", "r2i", "r3i", "r4i"), rec):
                    self.feed[k] = v
                out = composite(small, pha[0, 0].astype(np.float32), self.background(self.size), self.a)
                if out.shape[1::-1] != (self.a.width, self.a.height):
                    out = cv2.resize(out, (self.a.width, self.a.height), interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.result, self.seq = out, self.seq + 1
                n += 1
                if n == 30:
                    self.rate, n, t0 = 30 / (time.time() - t0), 0, time.time()
            except Exception as e:  # e.g. GPU device lost after suspend: rebuild the session
                print(f"matting error: {e!r}; rebuilding the session", flush=True)
                time.sleep(1)
                try:
                    self.load(self.loaded)
                except Exception as e2:
                    print(f"rebuild failed: {e2!r}", flush=True)
                    time.sleep(2)


class Recorder(threading.Thread):
    """Keeps the last few seconds of live output, JPEG-encoded, for motion pause.

    One entry per output frame, so playback runs at real speed; a repeated frame
    reuses the previous encoding instead of encoding it again.
    """

    def __init__(self, seconds, fps):
        super().__init__(daemon=True)
        self.frames = deque(maxlen=max(2, int(seconds * fps)))
        self.q = queue.Queue(maxsize=8)
        self.lock = threading.Lock()

    def add(self, frame):
        """`frame` is a new output frame, or None to repeat the previous one."""
        try:
            self.q.put_nowait(("frame", frame))
        except queue.Full:
            pass

    def clear(self):
        # Queued, so frames already waiting are dropped too; the buffer then
        # never holds a jump (a loop across one would show a visible cut).
        self.q.put(("clear", None))

    def snapshot(self):
        with self.lock:
            return list(self.frames)

    def run(self):
        while True:
            kind, f = self.q.get()
            try:
                with self.lock:
                    if kind == "clear":
                        self.frames.clear()
                        continue
                    if f is None:
                        if self.frames:
                            self.frames.append(self.frames[-1])
                        continue
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    with self.lock:
                        self.frames.append(buf)
            except Exception as e:
                print(f"recorder error: {e!r}", flush=True)


def cover(img, w, h):
    """Scale and centre-crop `img` to fill w x h without stretching."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return np.ascontiguousarray(img[y:y + h, x:x + w])


def blur_background(frame, alpha, radius):
    # Blur only background pixels (alpha-weighted) so the person doesn't
    # bleed into the blur as a halo. Done at quarter size for speed.
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
    w = 1.0 - cv2.resize(alpha, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    k = max(3, (radius // 4) * 2 + 1)
    num = cv2.GaussianBlur(small * w[..., None], (k, k), 0)
    den = cv2.GaussianBlur(w, (k, k), 0)[..., None]
    # Where no background is visible nearby, fall back to a plain blur.
    plain = cv2.GaussianBlur(small, (k, k), 0)
    bg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), plain)
    bg = bg.clip(0, 255).astype(np.uint8)
    return cv2.resize(bg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def composite(frame, alpha, bg_img, a):
    """`frame` over the background (`bg_img` at frame size, or a blur), at frame size."""
    if bg_img is not None:
        bg = bg_img
    elif a.blur > 0:
        bg = blur_background(frame, alpha, max(1, a.blur * frame.shape[0] // a.height))
    else:
        return frame
    return cv2.blendLinear(frame, bg, alpha, 1.0 - alpha)


def make_card(base, title, subtitle=None, dim=0.55):
    """A full frame: `base` darkened, with centred text."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = base.shape[:2]
    img = Image.fromarray(cv2.cvtColor((base * (1 - dim)).astype(np.uint8), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype(CARD_FONT, h // 10)
        small = ImageFont.truetype(CARD_FONT.replace("-Bold", "-Regular"), h // 28)
    except OSError:
        big = ImageFont.load_default(h // 10)
        small = ImageFont.load_default(h // 28)
    d.text((w / 2, h / 2), title, font=big, fill=(255, 255, 255), anchor="mm")
    if subtitle:
        d.text((w / 2, h / 2 + h // 10), subtitle, font=small, fill=(200, 200, 200), anchor="mm")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    a = parse_args()
    Control.limit = a.height
    Control.reload()
    # Before announcing the pid: SIGUSR1's default action would kill the process.
    signal.signal(signal.SIGUSR1, Control.reload)
    # Tells studio-cam this process understands SIGUSR1, and which newer features it has.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "pid").write_text(str(os.getpid()))
    (STATE_DIR / "caps").write_text(f"{os.getpid()} loop\n{os.getpid()} res\n")

    a.bg_img = None
    if a.background:
        img = cv2.imread(a.background)
        if img is None:
            raise SystemExit(f"cannot read {a.background}")
        a.bg_img = cover(img, a.width, a.height)
    bg_img = a.bg_img

    cam = Camera(a.camera, a.width, a.height, a.fps)
    cam.start()
    matter = Matter(cam, a)
    matter.start()
    rec = Recorder(a.loop_seconds, a.fps)
    rec.start()

    def card(title):
        """A text card over the background (or black), cached per title."""
        if title not in cards:
            cards[title] = make_card(bg_img if bg_img is not None else np.zeros((a.height, a.width, 3), np.uint8),
                                     title, dim=0.8)
        return cards[title]

    with pyvirtualcam.Camera(a.width, a.height, a.fps, device=a.out,
                             fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        n, t0, done = 0, time.time(), 0  # done: last matter.seq sent
        last, cards, held = None, {}, {}  # last: what was last sent; held: the frame each non-live mode shows
        shown, live_since = None, time.monotonic()  # shown: the non-live mode on screen, None when live
        clip, i, step, tick = [], 0, -1, 0
        while True:
            if not (cam.is_alive() and matter.is_alive() and rec.is_alive()):
                raise SystemExit("a worker thread died; exiting so studio-cam reports it")
            mode = Control.mode  # read once: the signal handler can change it at any moment
            if mode == "loop":
                if shown != "loop":
                    # Start at the newest frame going backward, so entering the loop is seamless.
                    clip, shown, tick = rec.snapshot(), "loop", 0
                    i, step = len(clip) - 1, -1
                if clip:
                    if tick % a.loop_hold == 0:
                        last = cv2.imdecode(clip[i], cv2.IMREAD_COLOR)
                        if len(clip) > 1:
                            if not 0 <= i + step < len(clip):
                                step = -step
                            i += step
                    tick += 1
                if last is not None:
                    vcam.send(last)
                vcam.sleep_until_next_frame()
                continue
            if mode != "live":
                # Built once per switch: paused freezes the last frame, camoff shows a card.
                if shown != mode:
                    if mode == "paused" and last is not None:
                        held[mode] = last.copy()  # freeze on the last frame viewers saw
                    else:
                        held[mode] = card("Paused" if mode == "paused" else "Camera off")
                    shown = mode
                last = held[mode]
                vcam.send(last)
                vcam.sleep_until_next_frame()
                continue
            if shown is not None:
                # Back to live: keep showing what was on screen (frozen frame, card or loop
                # frame) until a frame matted after resuming arrives.
                rec.clear()
                clip = []
                done = matter.seq
                shown, live_since = None, time.monotonic()
            with matter.lock:
                result, rseq = matter.result, matter.seq
            new = rseq != done
            if new:
                done = rseq
                last = result
                n += 1
            elif time.monotonic() - max(cam.last_time, live_since) > CAMERA_TIMEOUT:
                last = card("Camera unavailable")
            elif last is None:
                # Nothing matted yet (model loading): show the background, never the raw room.
                if bg_img is not None:
                    last = bg_img
                elif cam.frame is not None:
                    last = cv2.GaussianBlur(cv2.resize(cam.frame, (a.width, a.height)), (0, 0), 25)
            if last is not None:
                vcam.send(last)
                rec.add(last if new else None)
            vcam.sleep_until_next_frame()
            if a.stats and time.time() - t0 >= 2:
                print(f"output {n / (time.time() - t0):.1f} new frames/s, matte {matter.rate:.1f} fps", flush=True)
                n, t0 = 0, time.time()


if __name__ == "__main__":
    main()
