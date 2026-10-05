#!/usr/bin/env bash
# Set up Studio Cam (omarchy-cam-studio) on Omarchy / Arch Linux. Safe to re-run.
#   1. system packages (pacman; obs-shaderfilter-git from the AUR if yay/paru is present)
#   2. v4l2loopback config: /dev/video10 "Studio Cam" and /dev/video11 (matte feed)
#   3. Python venv and the matting models (downloaded and converted once, ~350 MB)
#   4. links: the studio-cam command, the app launcher entry, the Omarchy bar plugin
#   5. the OBS "Studio Cam" profile and scene collection (left alone if they exist)
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "System packages"
if command -v pacman >/dev/null; then
  # Only what's missing, so an existing OBS variant (obs-studio-git, ...) is left alone.
  need=()
  command -v obs >/dev/null || need+=(obs-studio)
  modinfo v4l2loopback >/dev/null 2>&1 || need+=(v4l2loopback-dkms)
  command -v v4l2-ctl >/dev/null || need+=(v4l-utils)
  command -v notify-send >/dev/null || need+=(libnotify)
  command -v uv >/dev/null || need+=(uv)
  if [ ${#need[@]} -gt 0 ]; then sudo pacman -S --needed --noconfirm "${need[@]}"; fi
  modinfo v4l2loopback >/dev/null 2>&1 ||
    echo "v4l2loopback did not build: install the headers for your kernel (e.g. linux-headers) and re-run"
  if ! pacman -Q obs-shaderfilter-git >/dev/null 2>&1; then
    aur=$(command -v yay || command -v paru || true)
    if [ -z "$aur" ] || ! "$aur" -S --needed --noconfirm obs-shaderfilter-git; then
      echo "obs-shaderfilter-git not installed: the optional 'Skin smoothing' filter will be missing"
    fi
  fi
else
  echo "No pacman: install OBS Studio, v4l2loopback, v4l-utils, libnotify and uv yourself"
fi

step "v4l2loopback devices"
install_conf() {  # install_conf SRC DEST: copy with sudo only if it differs
  cmp -s "$1" "$2" 2>/dev/null || sudo install -Dm644 "$1" "$2"
}
install_conf "$APP_DIR/system/v4l2loopback-modprobe.conf" /etc/modprobe.d/v4l2loopback.conf
install_conf "$APP_DIR/system/v4l2loopback-modules-load.conf" /etc/modules-load.d/v4l2loopback.conf
if [ ! -e /dev/video10 ] || [ ! -e /dev/video11 ]; then
  if grep -q '^v4l2loopback ' /proc/modules; then
    echo "v4l2loopback is loaded with other settings; reboot (or unload it) to get /dev/video10 and /dev/video11"
  else
    sudo modprobe v4l2loopback
  fi
fi

step "Python environment"
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  uv venv --python 3.13 "$APP_DIR/.venv"
fi
uv pip install --python "$APP_DIR/.venv/bin/python" -r "$APP_DIR/requirements.txt"

step "Matting models"
"$APP_DIR/.venv/bin/python" "$APP_DIR/tools/build_models.py"

step "Command, launcher and bar plugin"
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.config/omarchy/plugins"
ln -sfn "$APP_DIR/studio-cam" "$HOME/.local/bin/studio-cam"
install -m644 "$APP_DIR/studio-cam.desktop" "$HOME/.local/share/applications/studio-cam.desktop"
PLUGIN="$HOME/.config/omarchy/plugins/studio-cam"
if [ -d "$PLUGIN" ] && [ ! -L "$PLUGIN" ]; then
  echo "$PLUGIN is a real directory; move it away and re-run to link the plugin"
else
  ln -sfn "$APP_DIR/omarchy-plugin" "$PLUGIN"
fi

step "OBS profile and scene collection"
OBS="$HOME/.config/obs-studio/basic"
if [ -e "$OBS/profiles/Studio Cam/basic.ini" ]; then
  echo "OBS profile 'Studio Cam' exists, leaving it alone"
else
  install -Dm644 "$APP_DIR/obs/profile-basic.ini" "$OBS/profiles/Studio Cam/basic.ini"
fi
if [ -e "$OBS/scenes/Studio Cam.json" ]; then
  echo "OBS scene collection 'Studio Cam' exists, leaving it alone"
else
  install -Dm644 "$APP_DIR/obs/scene-collection.json" "$OBS/scenes/Studio Cam.json"
fi

step "Done"
cat <<EOF
Start it from the app launcher ("Studio Cam") or run: studio-cam
Then pick "Studio Cam" as the camera in your meeting app.

Optional:
  - Add the Studio Cam widget to your Omarchy bar (bar settings, Video category).
  - Put a background image at ~/.config/studio-cam/background.jpg (otherwise it blurs).
  - Run "studio-cam tune" to adjust the OBS filters or add a logo overlay.
EOF
