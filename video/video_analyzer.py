"""
============================================================
  Incident Report Analyzer — Student 4: Video Analyst
  Dataset : CAVIAR CCTV Dataset (University of Edinburgh)
  Output  : video/output/video_analysis.csv
            video/output/frames/  (annotated key frames)

  Run from the project root:
      python video/video_analyzer.py
============================================================
"""

# ── Cell 1: Imports & Install ─────────────────────────────
import sys
import os
import subprocess
import cv2
import csv
import uuid
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

# Auto-install missing libraries if needed
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "opencv-python", "numpy", "pandas", "--quiet"])
print("Libraries ready")


# ── Cell 2: Auto-detect Paths ─────────────────────────────
# Works on every machine after git clone — no hardcoded paths
BASE_DIR   = Path(__file__).resolve().parent   # video/
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
CSV_PATH   = OUTPUT_DIR / "video_analysis.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

print("Paths detected:")
print(f"   Data in   : {DATA_DIR}")
print(f"   CSV out   : {CSV_PATH}")
print(f"   Frames out: {FRAMES_DIR}")
print("All folders ready!")


# ── Cell 3: Generate Sample Videos ────────────────────────
# Skipped automatically if real .avi files already exist in data/

CLIPS = [
    "caviar_walk1_cam1.avi",
    "caviar_fight_seq.avi",
    "caviar_left_bag.avi",
    "caviar_fall_corridor.avi",
]

def make_synthetic_clip(filename, seed, fps=25, duration=5, width=320, height=240):
    path   = DATA_DIR / filename
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), False)
    rng    = random.Random(seed)
    for i in range(fps * duration):
        t     = i / fps
        frame = np.zeros((height, width), dtype=np.uint8)
        noise = np.random.randint(0, 20, (height, width), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        for _ in range(rng.randint(1, 3)):
            x = int(50 + 200 * abs(np.sin(t * 0.8 + rng.random())))
            y = int(40 + 150 * abs(np.cos(t * 0.6 + rng.random())))
            w = rng.randint(20, 60)
            h = rng.randint(30, 80)
            frame[y:y+h, x:x+w] = rng.randint(140, 240)
        cv2.line(frame, (0, height // 2), (width, height // 2), 60, 1)
        writer.write(frame)
    writer.release()
    print(f"   Created: {filename}")

existing = list(DATA_DIR.glob("*.avi")) + list(DATA_DIR.glob("*.mp4"))
if not existing:
    print("Generating synthetic CAVIAR-style CCTV clips...")
    for idx, name in enumerate(CLIPS):
        make_synthetic_clip(name, seed=idx * 17 + 3)
    print(f"\n{len(CLIPS)} video file(s) ready in {DATA_DIR}")
else:
    print(f"Found {len(existing)} existing video(s) in {DATA_DIR} — skipping generation")


# ── Cell 4: Video Analyzer ────────────────────────────────

# Settings
SAMPLE_EVERY_N_FRAMES  = 10
MOTION_THRESHOLD       = 500.0
CONFIDENCE_THRESHOLD   = 0.45
SAVE_FRAME_ON_INCIDENT = True

INCIDENT_RULES = [
    ({"fire", "smoke"},      "Fire",              "Critical", 0.70),
    ({"person", "fire"},     "Person in Fire",    "Critical", 0.65),
    ({"person", "knife"},    "Armed Threat",      "Critical", 0.70),
    ({"person", "gun"},      "Armed Threat",      "Critical", 0.70),
    ({"car", "truck"},       "Vehicle Crash",     "High",     0.60),
    ({"motorcycle", "car"},  "Vehicle Crash",     "High",     0.60),
    ({"person", "backpack"}, "Suspicious Person", "Medium",   0.55),
    ({"person"},             "Person Detected",   "Low",      0.50),
    ({"car"},                "Vehicle Moving",    "Low",      0.50),
]
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# CSV columns — exact match to assignment spec
CSV_HEADERS = [
    "Clip_ID",         # e.g. CAVIAR_WALK1
    "Timestamp",       # e.g. 00:00:12
    "Frame_ID",        # e.g. FRM_036
    "Event_Detected",  # e.g. Person collapsing
    "Persons_Count",   # e.g. 1 person
    "Confidence",      # e.g. 0.88
]


# Helper functions
def format_timestamp(frame_number, fps):
    secs    = int(frame_number / fps)
    h, rem  = divmod(secs, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_frame_id(frame_number):
    return f"FRM_{frame_number:03d}"

def clip_id_from_filename(filename):
    stem  = filename.replace(".avi", "").replace(".mp4", "").upper()
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem[:12]

def format_persons_count(detections):
    count = sum(1 for d in detections if d["label"].lower() == "person")
    if count == 0:
        return "none"
    return f"{count} person" if count == 1 else f"{count} persons"


# Object Detector (YOLOv8 with mock fallback)
class ObjectDetector:
    MOCK_POOL = [
        ("person", 0.92), ("car", 0.88), ("truck", 0.75),
        ("motorcycle", 0.80), ("backpack", 0.65),
        ("fire", 0.91), ("smoke", 0.85), ("knife", 0.70),
    ]
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("[Detector] YOLOv8 loaded")
        except Exception:
            print("[Detector] Mock mode — install ultralytics for real detections")

    def detect(self, frame, frame_number):
        if self.model:
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            return [{"label": r.names[int(b.cls)],
                     "confidence": round(float(b.conf), 3),
                     "bbox": [int(v) for v in b.xyxy[0]]}
                    for r in results for b in r.boxes]
        rng   = random.Random(frame_number * 7 + 42)
        picks = rng.sample(self.MOCK_POOL, min(rng.randint(0, 3), len(self.MOCK_POOL)))
        return [{"label": l, "confidence": c, "bbox": [50, 50, 200, 300]} for l, c in picks]


# Motion Analyzer (MOG2 background subtraction)
class MotionAnalyzer:
    def __init__(self):
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False)
    def score(self, frame):
        return float(np.count_nonzero(self.sub.apply(frame)))


# Incident Classifier
def classify(detections, motion):
    labels   = {d["label"].lower() for d in detections}
    avg_conf = (sum(d["confidence"] for d in detections) / len(detections)
                if detections else 0.0)
    btype, bsev, bconf = "No Incident", "Low", 0.0
    for req, itype, sev, mconf in INCIDENT_RULES:
        if req.issubset(labels) and avg_conf >= mconf:
            if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(bsev, 0):
                btype, bsev, bconf = itype, sev, avg_conf
    if btype == "No Incident" and motion > MOTION_THRESHOLD * 3:
        btype, bsev, bconf = "Suspicious Motion", "Medium", 0.55
    return btype, bsev, round(bconf, 3)


# Frame Annotator
def annotate(frame, detections, event, sev):
    out = frame.copy()
    col = {"Critical": (0,0,255), "High": (0,100,255),
           "Medium": (0,165,255), "Low": (0,200,0)}.get(sev, (180,180,180))
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        cv2.putText(out, f"{d['label']} {d['confidence']:.2f}",
                    (x1, max(y1-6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.rectangle(out, (0,0), (frame.shape[1], 28), (30,30,30), -1)
    cv2.putText(out, f"EVENT: {event}  [{sev}]",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


# Process one video file
def process_video(video_path, detector, motion_analyzer):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open: {video_path.name}")
        return []
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip  = clip_id_from_filename(video_path.name)
    print(f"\n  {video_path.name}  ({w}x{h}, {fps:.0f}fps, {total} frames)  Clip_ID: {clip}")

    rows, fidx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % SAMPLE_EVERY_N_FRAMES == 0:
            dets           = detector.detect(frame, fidx)
            motion         = motion_analyzer.score(frame)
            evt, sev, conf = classify(dets, motion)

            if SAVE_FRAME_ON_INCIDENT and evt not in ("No Incident", "Person Detected"):
                ann   = annotate(frame, dets, evt, sev)
                fname = f"{video_path.stem}_{format_frame_id(fidx)}_{evt.replace(' ','_')}.jpg"
                cv2.imwrite(str(FRAMES_DIR / fname), ann)

            rows.append({
                "Clip_ID":        clip,
                "Timestamp":      format_timestamp(fidx, fps),
                "Frame_ID":       format_frame_id(fidx),
                "Event_Detected": evt,
                "Persons_Count":  format_persons_count(dets),
                "Confidence":     conf,
            })

            if evt != "No Incident":
                print(f"    {format_frame_id(fidx)} | {format_timestamp(fidx, fps)} "
                      f"| {evt:<22} | {format_persons_count(dets):<10} | conf: {conf}")
        fidx += 1

    cap.release()
    incidents = sum(1 for r in rows if r["Event_Detected"] != "No Incident")
    print(f"  Done: {fidx} frames -> {len(rows)} samples, {incidents} incidents")
    return rows


# Main
print("=" * 60)
print("  Video Analyst — Student 4")
print(f"  Data   : {DATA_DIR}")
print(f"  Output : {CSV_PATH}")
print("=" * 60)

exts        = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
video_files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts]

if not video_files:
    print("No videos found in data/ folder.")
else:
    detector = ObjectDetector()
    motion   = MotionAnalyzer()
    all_rows = []

    for vf in sorted(video_files):
        all_rows.extend(process_video(vf, detector, motion))

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nCSV saved -> {CSV_PATH}  ({len(all_rows)} rows)")

    events = Counter(r["Event_Detected"] for r in all_rows)
    print("\nEvent breakdown:")
    for e, n in events.most_common():
        print(f"  {e:<25} {'#' * min(n, 35)} {n}")


# ── Cell 5: Display Results ───────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows    : {len(df)}")
    print(f"Videos        : {df['Clip_ID'].nunique()}")
    print(f"Incident types: {df['Event_Detected'].nunique()}")
    print()

    inc = df[df["Event_Detected"] != "No Incident"].copy().reset_index(drop=True)
    print(f"Incidents detected: {len(inc)}")
    print()
    print(inc[["Clip_ID", "Timestamp", "Frame_ID",
               "Event_Detected", "Persons_Count", "Confidence"]].to_string())
else:
    print("No CSV found.")
