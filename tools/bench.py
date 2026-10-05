"""Time matting at each processing size, the way matte.py runs it (prep + inference).

Usage: bench.py [gpu|cpu]   (default gpu). Close Studio Cam first for clean numbers.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matte  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "gpu"
frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)  # a 720p camera frame
for h, (w, _) in matte.PROC_SIZES.items():
    path = matte.HERE / "models" / f"rvm_resnet50_fp16_{w}x{h}_ds0.25.onnx"
    if not path.exists():
        print(f"{h}p: no model (run tools/build_models.py)")
        continue
    sess = matte.make_session(str(path), device, 8)
    feed = {i.name: np.zeros(i.shape, np.float16) for i in sess.get_inputs() if i.name != "src"}
    for i in range(40):
        if i == 10:
            t0 = time.time()
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        feed["src"] = cv2.dnn.blobFromImage(small, 1 / 255.0, swapRB=True).astype(np.float16)
        _, pha, *rec = sess.run(None, feed)
        feed.update(zip(("r1i", "r2i", "r3i", "r4i"), rec))
    ran_on = "gpu" if any("WebGpu" in p for p in sess.get_providers()) else "cpu"
    print(f"{h}p ({w}x{h}) {ran_on}: {30 / (time.time() - t0):.1f} fps", flush=True)
