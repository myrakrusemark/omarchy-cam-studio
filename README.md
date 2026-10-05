# omarchy-cam-studio

**Studio Cam**: an AI background-replacing virtual webcam for [Omarchy](https://omarchy.org) (Arch Linux + Hyprland), with a bar widget to pause, freeze or loop your video mid-meeting.

```
webcam ──► matte.py ──► /dev/video11 ──► OBS (color, skin smoothing, sharpen) ──► /dev/video10 "Studio Cam" ──► Meet / Teams / Zoom
          (AI matting,
           background swap)
```

- **Background replacement or blur** using [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) on the GPU (onnxruntime's WebGPU provider, so it runs on integrated AMD/Intel graphics through Vulkan).
- **Adjustable processing size** (360p / 480p / 720p) that switches live. Video and mask are processed at that size and upscaled, so the stream stays 720p and never drops; lower is faster and a little softer.
- **The mask always matches the frame.** Every frame sent is composited with the mask computed from that same frame, so the cutout never trails behind you when you move.
- **Pause** freezes the last frame, **motion pause** loops the last 30 seconds back and forth so you still look live, and **camera off** shows a card and turns the webcam light off. The virtual camera keeps streaming in every mode, so meeting apps never lose it.

## Install

On Omarchy / Arch:

```sh
git clone https://github.com/myrakrusemark/omarchy-cam-studio.git
cd omarchy-cam-studio
./install.sh
```

`install.sh` is safe to re-run. It:

1. installs `obs-studio`, `v4l2loopback-dkms`, `v4l-utils`, `libnotify` and `uv` (plus `obs-shaderfilter-git` from the AUR if `yay` or `paru` is available; it provides the optional skin-smoothing filter),
2. configures v4l2loopback to create `/dev/video10` ("Studio Cam") and `/dev/video11` (the matte feed) at boot,
3. creates `.venv/` and builds the matting models in `models/` (downloads the 107 MB ResNet-50 RVM model once, then converts it to fixed-size fp16 models),
4. links the `studio-cam` command into `~/.local/bin`, adds a launcher entry, and links the bar plugin into `~/.config/omarchy/plugins/studio-cam`,
5. creates the OBS "Studio Cam" profile and scene collection (an existing one is left alone).

Then add the **Studio Cam** widget to your bar (Omarchy bar settings, Video category).

If v4l2loopback was already loaded with other settings, reboot once so the devices come up as `/dev/video10` and `/dev/video11`.

## Use

Start it from the launcher or the bar widget, then choose **Studio Cam** as the camera in your meeting app.

| Command | What it does |
|---|---|
| `studio-cam` | start if stopped, quit if running |
| `studio-cam start` / `quit` | start or stop the pipeline |
| `studio-cam pause` | toggle a frozen frame |
| `studio-cam loop` | toggle motion pause (the last 30 s, back and forth) |
| `studio-cam camoff` | toggle a "Camera off" card (releases the webcam) |
| `studio-cam live` | back to normal video |
| `studio-cam res 360\|480\|720` | processing size, applied live (no argument prints it) |
| `studio-cam status` | `off`, `starting`, `live`, `paused`, `loop`, `camoff`, or `noobs` (OBS is not running) |
| `studio-cam tune` | start with OBS visible to adjust filters |

Middle-clicking the bar icon pauses or resumes.

## Configure

Everything personal lives in `~/.config/studio-cam/`, outside the repo:

- `background.jpg` (or `.png`): the replacement background. Without it, the real background is blurred.
- `resolution`: the processing size, written by `studio-cam res`.

Environment variables:

- `STUDIO_CAM_CAMERA`: the webcam device. By default, the first `/dev/v4l/by-id/*-video-index0`.
- `STUDIO_CAM_ARGS`: extra `matte.py` options, e.g. `--blur 40` or `--loop-seconds 20` (see `matte.py --help`).

Filters and overlays (a logo, for example) are ordinary OBS settings: run `studio-cam tune`.

Logs go to `~/.cache/studio-cam.log`.

## Layout

| Path | |
|---|---|
| `studio-cam` | the CLI; starts `matte.py` and OBS, switches modes by signalling `matte.py` |
| `matte.py` | camera capture, matting, compositing, and the virtual camera feed |
| `omarchy-plugin/` | the Quickshell bar widget |
| `obs/` | OBS profile and scene collection templates |
| `system/` | v4l2loopback modprobe and modules-load config |
| `tools/build_models.py` | downloads RVM and builds the fixed-size fp16 models |
| `tools/bench.py` | matting speed at each processing size |

## Credits and license

Matting uses [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) by Shanchuan Lin et al. Its model weights are downloaded at install time and are not part of this repository; they are under RVM's own license (GPL-3.0).

The code in this repository is MIT licensed.
