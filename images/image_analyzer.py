"""
Student 3 — Image Analyst
Processes crime scene / accident photographs using YOLOv8 for object detection,
pytesseract for OCR, and HuggingFace for scene classification.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Uncomment these imports as you implement each stage ──
# from ultralytics import YOLO
# import cv2
# import pytesseract
# from PIL import Image


class ImagePipeline(BasePipeline):
    """Image analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "image"
    SOURCE_LABEL = "image"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Image_ID", "Scene_Type",
        "Objects_Detected", "Bounding_Boxes", "Text_Extracted",
        "Confidence", "Severity",
    ]

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load image files (.jpg, .png) from data directory."""
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.raw_data = []

        for ext in image_extensions:
            self.raw_data.extend(glob.glob(os.path.join(self.data_dir, ext)))

        self.logger.info(f"Found {len(self.raw_data)} image files")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Run object detection and OCR on images."""
        self.processed_data = []

        # TODO: Implement YOLOv8 object detection
        # model = YOLO("yolov8n.pt")  # or your fine-tuned model
        #
        # for img_path in self.raw_data:
        #     results = model(img_path)
        #
        #     # Extract detections
        #     detections = []
        #     for result in results:
        #         for box in result.boxes:
        #             detections.append({
        #                 "class": result.names[int(box.cls[0])],
        #                 "confidence": float(box.conf[0]),
        #                 "bbox": box.xyxy[0].tolist(),
        #             })
        #
        #     # Run OCR for visible text
        #     ocr_text = pytesseract.image_to_string(Image.open(img_path))
        #
        #     self.processed_data.append({
        #         "file": img_path,
        #         "detections": detections,
        #         "ocr_text": ocr_text.strip(),
        #     })

        self.logger.info(f"Processed {len(self.processed_data)} images")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from detection results."""
        self.extracted_records = []

        # TODO: Implement extraction logic
        # for idx, item in enumerate(self.processed_data, start=1):
        #     objects = [d["class"] for d in item["detections"]]
        #     avg_conf = (
        #         sum(d["confidence"] for d in item["detections"]) / len(item["detections"])
        #         if item["detections"] else 0.0
        #     )
        #     bbox_summary = f"{len(item['detections'])} detections"
        #
        #     # Classify scene type based on detected objects
        #     scene = "Unknown"
        #     if "fire" in objects or "smoke" in objects:
        #         scene = "Fire Scene"
        #     elif "car" in objects or "truck" in objects:
        #         scene = "Accident Scene"
        #     elif "person" in objects and "knife" in objects:
        #         scene = "Crime Scene"
        #
        #     record = {
        #         "Incident_ID": generate_incident_id(idx),
        #         "Source": self.SOURCE_LABEL,
        #         "Image_ID": f"IMG_{idx:03d}",
        #         "Scene_Type": scene,
        #         "Objects_Detected": ", ".join(set(objects)),
        #         "Bounding_Boxes": bbox_summary,
        #         "Text_Extracted": item["ocr_text"][:200],
        #         "Confidence": round(avg_conf, 2),
        #         "Severity": classify_severity(scene + " " + " ".join(objects), avg_conf),
        #     }
        #     self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = ImagePipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
