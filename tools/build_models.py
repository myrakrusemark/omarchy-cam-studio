"""Download Robust Video Matting and build the fixed-size fp16 models matte.py uses.

For each processing size, bakes the input size and downsample ratio into the
model (static shapes let the GPU execution provider plan ahead instead of
re-allocating every frame), simplifies it, and converts it to fp16.
Writes models/rvm_resnet50_fp16_{W}x{H}_ds0.25.onnx; existing ones are kept.

Usage: build_models.py [HEIGHT ...]   (default: 360 480 720)
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxconverter_common import float16
from onnxsim import simplify

MODELS = Path(__file__).resolve().parent.parent / "models"
BASE = MODELS / "rvm_resnet50_fp32.onnx"
BASE_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx"
BASE_SHA256 = "25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c"
SIZES = {360: (640, 360), 480: (854, 480), 720: (1280, 720)}  # keep in sync with matte.PROC_SIZES
RATIO = 0.25


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_base():
    if BASE.exists():
        if sha256(BASE) == BASE_SHA256:
            return
        print(f"{BASE.name} has the wrong checksum; downloading it again", flush=True)
        BASE.unlink()
    MODELS.mkdir(exist_ok=True)
    print(f"downloading {BASE_URL}", flush=True)
    tmp = BASE.with_suffix(".part")
    urllib.request.urlretrieve(BASE_URL, tmp)
    digest = sha256(tmp)
    if digest != BASE_SHA256:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for {BASE.name}: {digest}")
    tmp.rename(BASE)


def build(w, h, ds):
    out = MODELS / f"rvm_resnet50_fp16_{w}x{h}_ds{ds}.onnx"
    if out.exists():
        print(f"{out.name} exists", flush=True)
        return
    # Learn the recurrent-state shapes for this size by running once on CPU.
    sess = ort.InferenceSession(str(BASE), providers=["CPUExecutionProvider"])
    z = np.zeros((1, 1, 1, 1), np.float32)
    outs = sess.run(None, {"src": np.zeros((1, 3, h, w), np.float32), "r1i": z, "r2i": z, "r3i": z,
                           "r4i": z, "downsample_ratio": np.array([ds], np.float32)})
    shapes = {"src": (1, 3, h, w), **{f"r{i + 1}i": o.shape for i, o in enumerate(outs[2:])}}

    m = onnx.load(str(BASE))
    g = m.graph
    for inp in list(g.input):
        if inp.name == "downsample_ratio":
            g.input.remove(inp)
            g.initializer.append(numpy_helper.from_array(np.array([ds], np.float32), "downsample_ratio"))
            continue
        for d, v in zip(inp.type.tensor_type.shape.dim, shapes[inp.name]):
            d.ClearField("dim_param")
            d.dim_value = v
    for o in g.output:
        del o.type.tensor_type.shape.dim[:]
    m, ok = simplify(m)
    if not ok:
        raise SystemExit(f"onnxsim could not validate the {w}x{h} model")
    m = float16.convert_float_to_float16(m, keep_io_types=False)
    tmp = out.with_suffix(".part")
    onnx.save(m, str(tmp))
    tmp.rename(out)
    print(f"built {out.name}", flush=True)


def main():
    heights = [int(a) for a in sys.argv[1:]] or list(SIZES)
    unknown = [h for h in heights if h not in SIZES]
    if unknown:
        raise SystemExit(f"unknown size(s) {unknown}; choose from {list(SIZES)}")
    download_base()
    for h in heights:
        build(*SIZES[h], RATIO)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", module="onnxconverter_common")  # fp16 truncation chatter
    main()
