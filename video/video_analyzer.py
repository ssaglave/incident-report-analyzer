"""
Student 4 — Video Analyst
Processes CCTV / surveillance footage using OpenCV for frame extraction,
YOLOv8 for object detection, and motion analysis for anomaly detection.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Uncomment these imports as you implement each stage ──
# import cv2
# from ultralytics import YOLO
# import numpy as np


class VideoPipeline(BasePipeline):
    """Video analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "video"
    SOURCE_LABEL = "video"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Clip_ID", "Timestamp",
        "Frame_ID", "Event_Detected", "Persons_Count",
        "Confidence", "Severity",
    ]

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load video files (.mp4, .mpg, .avi) from data directory."""
        video_extensions = ("*.mp4", "*.mpg", "*.avi", "*.mov")
        self.raw_data = []

        for ext in video_extensions:
            self.raw_data.extend(glob.glob(os.path.join(self.data_dir, ext)))

        self.logger.info(f"Found {len(self.raw_data)} video files")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Extract frames and run object detection + motion analysis."""
        self.processed_data = []

        # TODO: Implement frame extraction and detection
        # model = YOLO("yolov8n.pt")
        #
        # for video_path in self.raw_data:
        #     cap = cv2.VideoCapture(video_path)
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     frame_interval = max(int(fps), 1)  # 1 frame per second
        #     frame_count = 0
        #     prev_frame_gray = None
        #
        #     while cap.isOpened():
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #
        #         if frame_count % frame_interval == 0:
        #             # Object detection
        #             results = model(frame)
        #
        #             # Motion detection via frame differencing
        #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #             motion_score = 0.0
        #             if prev_frame_gray is not None:
        #                 diff = cv2.absdiff(prev_frame_gray, gray)
        #                 motion_score = float(np.mean(diff)) / 255.0
        #             prev_frame_gray = gray
        #
        #             # Count persons
        #             persons = sum(1 for box in results[0].boxes
        #                           if results[0].names[int(box.cls[0])] == "person")
        #
        #             timestamp = frame_count / fps
        #             minutes = int(timestamp // 60)
        #             seconds = int(timestamp % 60)
        #
        #             self.processed_data.append({
        #                 "video_file": video_path,
        #                 "frame_number": frame_count,
        #                 "timestamp": f"00:{minutes:02d}:{seconds:02d}",
        #                 "detections": results,
        #                 "persons_count": persons,
        #                 "motion_score": motion_score,
        #             })
        #
        #             # Save key frame
        #             frame_path = os.path.join(self.output_dir, "frames",
        #                                       f"frame_{frame_count:05d}.jpg")
        #             cv2.imwrite(frame_path, frame)
        #
        #         frame_count += 1
        #     cap.release()

        self.logger.info(f"Processed {len(self.processed_data)} frames")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from video frame analysis."""
        self.extracted_records = []

        # TODO: Implement extraction logic
        # for idx, item in enumerate(self.processed_data, start=1):
        #     # Classify event based on motion + detections
        #     event = "Normal activity"
        #     if item["motion_score"] > 0.3:
        #         event = "Significant motion detected"
        #     if item["persons_count"] == 0 and item["motion_score"] > 0.5:
        #         event = "Object movement (no persons)"
        #
        #     clip_name = os.path.basename(item["video_file"]).split(".")[0]
        #
        #     record = {
        #         "Incident_ID": generate_incident_id(idx),
        #         "Source": self.SOURCE_LABEL,
        #         "Clip_ID": clip_name,
        #         "Timestamp": item["timestamp"],
        #         "Frame_ID": f"FRM_{item['frame_number']:05d}",
        #         "Event_Detected": event,
        #         "Persons_Count": item["persons_count"],
        #         "Confidence": round(item["motion_score"], 2),
        #         "Severity": classify_severity(event, item["motion_score"]),
        #     }
        #     self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = VideoPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
