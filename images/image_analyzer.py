"""
Student 3 - Image Analyst
Processes incident photographs using:
  - Stage 1: Load image files from images/data
  - Stage 2: YOLOv8 object detection + pytesseract OCR
  - Stage 3: Scene classification + structured CSV output

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob
import shutil
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(os.path.dirname(__file__), ".ultralytics"),
)

from pipeline import BasePipeline, generate_incident_id, classify_severity

from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


SCENE_RULES = {
    "Fire Scene": {"fire", "smoke", "flame"},
    "Accident Scene": {"car", "truck", "bus", "motorcycle", "bicycle"},
    "Crime Scene": {"person", "knife", "gun", "weapon"},
    "Emergency Response Scene": {"ambulance", "fire hydrant", "traffic light"},
}

OCR_HIGH_SEVERITY_HINTS = (
    "fire",
    "help",
    "danger",
    "emergency",
    "hazard",
    "warning",
    "evacuate",
)

MAX_ANNOTATED_IMAGES = 100


class ImagePipeline(BasePipeline):
    """Image analysis pipeline - plugs into the common 5-stage framework."""

    MODULE_NAME = "image"
    SOURCE_LABEL = "image"
    OUTPUT_COLUMNS = [
        "Incident_ID",
        "Source",
        "Image_ID",
        "Scene_Type",
        "Objects_Detected",
        "Bounding_Boxes",
        "Text_Extracted",
        "Confidence",
        "Severity",
    ]

    def __init__(self, data_dir: str, output_dir: str, model_path: str = None):
        super().__init__(data_dir, output_dir)
        default_model = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")
        self.model_path = model_path or default_model
        self.detector = None
        self.annotated_dir = os.path.join(output_dir, "annotated")
        self.class_names = {}
        self._ocr_available = None
        self.annotated_saved = 0

    # Stage 1: Data Ingestion
    def load_data(self):
        """Load image files from the data directory recursively."""
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.raw_data = []

        for ext in image_extensions:
            pattern = os.path.join(self.data_dir, "**", ext)
            self.raw_data.extend(glob.glob(pattern, recursive=True))

        self.raw_data = sorted(set(self.raw_data))
        self.logger.info(f"Found {len(self.raw_data)} image files")

        if not self.raw_data:
            self.logger.warning(
                "No images found. Add your Roboflow dataset images to images/data/"
            )

    # Stage 2: AI Processing
    def process_data(self):
        """Run YOLO object detection and OCR on images."""
        self.processed_data = []

        if not self.raw_data:
            self.logger.info("Skipping processing because no input images were found.")
            return

        self.class_names = self._load_dataset_class_names()
        self._load_models()
        os.makedirs(self.annotated_dir, exist_ok=True)

        for img_path in self.raw_data:
            self.logger.info(f"Processing image: {os.path.basename(img_path)}")
            detections = self._load_label_detections(img_path)
            if not detections:
                detections = self._detect_objects(img_path)
            ocr_text = self._extract_text(img_path)
            annotated_path = self._save_annotated_image(img_path, detections)

            self.processed_data.append(
                {
                    "file": img_path,
                    "detections": detections,
                    "ocr_text": ocr_text,
                    "annotated_path": annotated_path,
                }
            )

        self.logger.info(f"Processed {len(self.processed_data)} images")

    def _load_models(self):
        """Load the detection model if ultralytics is installed."""
        if self.detector is not None:
            return

        if YOLO is None:
            self.logger.warning(
                "ultralytics is not installed. Object detection will be skipped."
            )
            return

        try:
            self.detector = YOLO(self.model_path)
            self.logger.info(f"Loaded YOLO model: {self.model_path}")
        except Exception as exc:
            self.logger.warning(f"Could not load YOLO model '{self.model_path}': {exc}")
            self.detector = None

    def _load_dataset_class_names(self) -> Dict[int, str]:
        """Read class names from a YOLO data.yaml file when present."""
        yaml_path = os.path.join(self.data_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            return {}

        names = {}
        try:
            with open(yaml_path, "r", encoding="utf-8") as handle:
                lines = [line.rstrip() for line in handle]

            in_names = False
            idx = 0
            for line in lines:
                stripped = line.strip()
                if stripped == "names:":
                    in_names = True
                    continue
                if in_names:
                    if not stripped.startswith("- "):
                        break
                    value = stripped[2:].strip().strip("'").strip('"')
                    names[idx] = value or f"class_{idx}"
                    idx += 1
        except Exception as exc:
            self.logger.warning(f"Could not parse data.yaml class names: {exc}")

        return names

    def _load_label_detections(self, img_path: str) -> List[Dict]:
        """Load detections from YOLO label files bundled with the dataset."""
        label_path = self._guess_label_path(img_path)
        if not label_path or not os.path.exists(label_path):
            return []

        try:
            with Image.open(img_path) as image:
                width, height = image.size
        except Exception:
            return []

        detections = []
        try:
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    x1 = round(x_center - (box_width / 2), 2)
                    y1 = round(y_center - (box_height / 2), 2)
                    x2 = round(x_center + (box_width / 2), 2)
                    y2 = round(y_center + (box_height / 2), 2)
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    if class_name == "0":
                        class_name = "unknown"
                    detections.append(
                        {
                            "class": class_name,
                            "confidence": 1.0,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
        except Exception as exc:
            self.logger.warning(
                f"Could not read label file for {os.path.basename(img_path)}: {exc}"
            )
            return []

        return detections

    def _guess_label_path(self, img_path: str) -> str:
        """Map an image path inside a YOLO dataset to its label file."""
        label_dir = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        stem, _ = os.path.splitext(label_dir)
        return f"{stem}.txt"

    def _detect_objects(self, img_path: str) -> List[Dict]:
        """Run YOLO detection and return a normalized detection list."""
        if self.detector is None:
            return []

        detections = []

        try:
            results = self.detector(img_path, verbose=False)
            for result in results:
                names = result.names
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    detections.append(
                        {
                            "class": names[class_id],
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
        except Exception as exc:
            self.logger.warning(
                f"Object detection failed for {os.path.basename(img_path)}: {exc}"
            )

        return detections

    def _extract_text(self, img_path: str) -> str:
        """Run OCR if pytesseract is available."""
        if pytesseract is None:
            if self._ocr_available is not False:
                self.logger.warning("pytesseract is not installed. OCR will be skipped.")
                self._ocr_available = False
            return ""

        if self._ocr_available is None:
            self._ocr_available = shutil.which("tesseract") is not None
            if not self._ocr_available:
                self.logger.warning(
                    "Tesseract is not installed or not in PATH. OCR will be skipped."
                )

        if not self._ocr_available:
            return ""

        try:
            text = pytesseract.image_to_string(Image.open(img_path))
            return " ".join(text.split())
        except Exception as exc:
            self.logger.warning(
                f"OCR failed for {os.path.basename(img_path)}: {exc}"
            )
            return ""

    def _save_annotated_image(self, img_path: str, detections: List[Dict]) -> str:
        """Save an annotated copy of the image with detected boxes."""
        if not detections or self.annotated_saved >= MAX_ANNOTATED_IMAGES:
            return ""

        try:
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class']} {det['confidence']:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1 + 4, max(y1 - 18, 0)), label, fill="red")

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            annotated_path = os.path.join(self.annotated_dir, f"{base_name}_annotated.jpg")
            image.save(annotated_path)
            self.annotated_saved += 1
            return annotated_path
        except Exception as exc:
            self.logger.warning(
                f"Could not save annotated image for {os.path.basename(img_path)}: {exc}"
            )
            return ""

    # Stage 3: Information Extraction
    def extract_information(self):
        """Convert detection and OCR output into the shared CSV schema."""
        self.extracted_records = []

        for idx, item in enumerate(self.processed_data, start=1):
            detections = item["detections"]
            object_names = [det["class"] for det in detections]
            scene_type = self._classify_scene(object_names, item["ocr_text"])
            confidence = self._average_confidence(detections)
            bbox_summary = self._summarize_boxes(detections)
            severity = self._classify_image_severity(
                scene_type=scene_type,
                object_names=object_names,
                ocr_text=item["ocr_text"],
                confidence=confidence,
            )

            record = {
                "Incident_ID": generate_incident_id(idx),
                "Source": self.SOURCE_LABEL,
                "Image_ID": f"IMG_{idx:03d}",
                "Scene_Type": scene_type,
                "Objects_Detected": ", ".join(dict.fromkeys(object_names)),
                "Bounding_Boxes": bbox_summary,
                "Text_Extracted": item["ocr_text"][:250],
                "Confidence": round(confidence, 2),
                "Severity": severity,
            }
            self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")

    def _classify_scene(self, object_names: List[str], ocr_text: str) -> str:
        """Choose a scene label from detected objects and OCR hints."""
        object_set = {name.lower() for name in object_names}

        for scene_name, keywords in SCENE_RULES.items():
            if object_set.intersection(keywords):
                return scene_name

        ocr_lower = ocr_text.lower()
        if any(term in ocr_lower for term in ("fire", "smoke", "alarm")):
            return "Fire Scene"
        if any(term in ocr_lower for term in ("police", "crime", "suspect")):
            return "Crime Scene"
        if object_names:
            return "General Incident Scene"
        return "Unknown"

    def _average_confidence(self, detections: List[Dict]) -> float:
        """Calculate mean confidence across detections."""
        if not detections:
            return 0.0
        return sum(det["confidence"] for det in detections) / len(detections)

    def _summarize_boxes(self, detections: List[Dict]) -> str:
        """Summarize detection classes and their bounding boxes."""
        if not detections:
            return ""

        parts = []
        for det in detections[:10]:
            x1, y1, x2, y2 = det["bbox"]
            parts.append(f"{det['class']}:[{x1},{y1},{x2},{y2}]")
        return "; ".join(parts)

    def _classify_image_severity(
        self,
        scene_type: str,
        object_names: List[str],
        ocr_text: str,
        confidence: float,
    ) -> str:
        """Apply image-specific severity rules and fall back to the shared helper."""
        text_blob = " ".join([scene_type, " ".join(object_names), ocr_text]).strip()
        lowered_objects = {name.lower() for name in object_names}
        ocr_lower = ocr_text.lower()

        if {"fire", "smoke", "gun", "knife"}.intersection(lowered_objects):
            return "High"
        if any(term in ocr_lower for term in OCR_HIGH_SEVERITY_HINTS):
            return "High"
        if {"car", "truck", "bus"}.intersection(lowered_objects):
            return "Medium" if confidence >= 0.4 else "Low"

        return classify_severity(text_blob, confidence)


if __name__ == "__main__":
    pipeline = ImagePipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
