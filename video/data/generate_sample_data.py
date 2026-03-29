"""
Generate synthetic CAVIAR-style CCTV clips for testing.
Run from anywhere — paths are relative to this file's location.

Usage:
    python video/data/generate_sample_data.py
"""

import cv2
import numpy as np
import random
from pathlib import Path

OUT_DIR  = Path(__file__).resolve().parent   # video/data/
FPS      = 25
DURATION = 5
W, H     = 320, 240

CLIPS = [
    "caviar_walk1_cam1.avi",
    "caviar_fight_seq.avi",
    "caviar_left_bag.avi",
    "caviar_fall_corridor.avi",
]

def generate(name, seed):
    path   = OUT_DIR / name
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H), False)
    rng    = random.Random(seed)
    for i in range(FPS * DURATION):
        t     = i / FPS
        frame = np.zeros((H, W), dtype=np.uint8)
        noise = np.random.randint(0, 20, (H, W), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        for _ in range(rng.randint(1, 3)):
            x = int(50 + 200 * abs(np.sin(t * 0.8 + rng.random())))
            y = int(40 + 150 * abs(np.cos(t * 0.6 + rng.random())))
            w = rng.randint(20, 60)
            h = rng.randint(30, 80)
            frame[y:y+h, x:x+w] = rng.randint(140, 240)
        cv2.line(frame, (0, H//2), (W, H//2), 60, 1)
        writer.write(frame)
    writer.release()
    print(f"  Created: {name}")

if __name__ == "__main__":
    print(f"Generating clips in {OUT_DIR} ...")
    for idx, name in enumerate(CLIPS):
        generate(name, seed=idx * 17 + 3)
    print("Done. Run: python video/video_analyzer.py")
