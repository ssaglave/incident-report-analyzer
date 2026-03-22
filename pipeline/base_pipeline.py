"""
Multimodal Incident Report Analyzer — Common AI Pipeline

This is the SHARED pipeline that ALL 5 students use.
Each student only implements their modality-specific processor by subclassing BasePipeline.

Pipeline Stages:
    Stage 1: Unstructured Data Ingestion   → load_data()
    Stage 2: AI Processing per Modality    → process_data()
    Stage 3: Information Extraction        → extract_information()
    Stage 4: Structured Dataset Generation → generate_output()
    Stage 5: Dashboard / Query System      → (handled by dashboard/app.py)

Usage:
    See audio/audio_analyzer.py, pdf/pdf_analyzer.py, etc. for examples.
"""

import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-12s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)


class BasePipeline(ABC):
    """
    Abstract base class for the common AI pipeline.

    Every student subclasses this and implements:
        - load_data()            → Stage 1: Ingestion
        - process_data()         → Stage 2: AI Processing
        - extract_information()  → Stage 3: Extraction

    Stage 4 (Structured Output) and Stage 5 (Dashboard) are handled
    by the base class automatically.
    """

    # ── Must be set by each student's subclass ──────────────────────────────

    MODULE_NAME = ""        # e.g., "audio", "pdf", "image", "video", "text"
    SOURCE_LABEL = ""       # value for the 'Source' column in CSV
    OUTPUT_COLUMNS = []     # list of column names for the output CSV

    def __init__(self, data_dir: str, output_dir: str):
        """
        Args:
            data_dir:   Path to the raw data folder   (e.g., "audio/data")
            output_dir: Path to the output folder      (e.g., "audio/output")
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.MODULE_NAME or self.__class__.__name__)

        # Internal storage passed between stages
        self.raw_data = None        # Stage 1 output
        self.processed_data = None  # Stage 2 output
        self.extracted_records = [] # Stage 3 output (list of dicts)
        self.output_df = None       # Stage 4 output (DataFrame)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: UNSTRUCTURED DATA INGESTION
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def load_data(self):
        """
        Load raw unstructured data from self.data_dir.

        Each student implements this to handle their file type:
            - Student 1 (Audio):  Load .wav/.mp3 files
            - Student 2 (PDF):    Load .pdf files
            - Student 3 (Image):  Load .jpg/.png files
            - Student 4 (Video):  Load .mpg/.mp4 files
            - Student 5 (Text):   Load .csv/.txt files

        Must populate: self.raw_data
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: AI PROCESSING PER MODALITY
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def process_data(self):
        """
        Run AI model(s) on the ingested data.

        Each student implements this with their specific AI tools:
            - Student 1: Whisper (speech-to-text)
            - Student 2: pdfplumber / pytesseract (PDF parsing / OCR)
            - Student 3: YOLOv8 (object detection)
            - Student 4: OpenCV + YOLOv8 (frame extraction + detection)
            - Student 5: spaCy + HuggingFace (NER + sentiment)

        Input:  self.raw_data (from Stage 1)
        Must populate: self.processed_data
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3: INFORMATION EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def extract_information(self):
        """
        Extract structured fields from AI-processed results.

        Each student implements this to pull out key fields:
            - Event type, location, time, entities, sentiment, severity

        Input:  self.processed_data (from Stage 2)
        Must populate: self.extracted_records (list of dicts)

        Each dict in extracted_records MUST include:
            - 'Incident_ID': str  (e.g., 'INC_001')
            - 'Source': str       (e.g., 'audio')
            - 'Severity': str     ('Low' / 'Medium' / 'High')
            + all module-specific columns
        """
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 4: STRUCTURED DATASET GENERATION  (Common — handled by base class)
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_output(self):
        """
        Convert extracted records into a structured CSV file.
        This is COMMON logic — students do NOT override this.
        """
        self.logger.info("Stage 4: Generating structured output...")

        if not self.extracted_records:
            self.logger.warning("No records extracted. Output CSV will be empty.")
            self.output_df = pd.DataFrame(columns=self.OUTPUT_COLUMNS)
        else:
            self.output_df = pd.DataFrame(self.extracted_records)

        # Ensure required columns exist
        required = ["Incident_ID", "Source", "Severity"]
        for col in required:
            if col not in self.output_df.columns:
                self.logger.warning(f"Missing required column: {col}")

        # Validate Severity values
        valid_severities = {"Low", "Medium", "High", ""}
        if "Severity" in self.output_df.columns:
            invalid = self.output_df[~self.output_df["Severity"].isin(valid_severities)]
            if len(invalid) > 0:
                self.logger.warning(
                    f"Found {len(invalid)} rows with invalid Severity values. "
                    f"Allowed: Low, Medium, High"
                )

        # Fill missing values with empty string (per schema rules)
        self.output_df = self.output_df.fillna("")

        # Save to CSV
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.MODULE_NAME}_results.csv")
        self.output_df.to_csv(output_path, index=False, encoding="utf-8")

        self.logger.info(f"Saved {len(self.output_df)} records → {output_path}")
        return output_path

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self):
        """
        Execute the full pipeline: Stage 1 → Stage 2 → Stage 3 → Stage 4.
        Stage 5 (Dashboard) is handled separately by dashboard/app.py.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"  PIPELINE START: {self.MODULE_NAME.upper()} MODULE")
        self.logger.info("=" * 60)

        start_time = datetime.now()

        # Stage 1: Ingest
        self.logger.info("Stage 1: Loading unstructured data...")
        self.load_data()
        self.logger.info(f"Stage 1 complete. Data loaded from: {self.data_dir}")

        # Stage 2: AI Processing
        self.logger.info("Stage 2: Running AI processing...")
        self.process_data()
        self.logger.info("Stage 2 complete. AI processing done.")

        # Stage 3: Extraction
        self.logger.info("Stage 3: Extracting structured information...")
        self.extract_information()
        self.logger.info(f"Stage 3 complete. Extracted {len(self.extracted_records)} records.")

        # Stage 4: Output
        output_path = self.generate_output()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info("=" * 60)
        self.logger.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info("=" * 60)

        return self.output_df


# ─── Utility: Generate Incident IDs ──────────────────────────────────────────

def generate_incident_id(index: int) -> str:
    """Generate a standardized Incident ID. Example: INC_001"""
    return f"INC_{index:03d}"


# ─── Utility: Classify Severity ──────────────────────────────────────────────

HIGH_KEYWORDS = [
    "fire", "trapped", "shooting", "weapon", "gun", "knife", "murder",
    "homicide", "explosion", "collapse", "critical", "death", "fatal",
    "stabbing", "hostage", "bomb",
]

MEDIUM_KEYWORDS = [
    "theft", "robbery", "accident", "crash", "assault", "burglary",
    "vandalism", "fight", "injury", "stolen", "break-in",
]


def classify_severity(text: str, confidence: float = 0.0) -> str:
    """
    Rule-based severity classification.
    Students can use this or implement their own logic.

    Args:
        text:       Any text describing the event
        confidence: Optional confidence score (0.0–1.0)

    Returns:
        'High', 'Medium', or 'Low'
    """
    if not text:
        return "Low"

    text_lower = text.lower()

    # Check high-severity keywords
    for keyword in HIGH_KEYWORDS:
        if keyword in text_lower:
            return "High"

    # Check medium-severity keywords
    for keyword in MEDIUM_KEYWORDS:
        if keyword in text_lower:
            return "Medium"

    # Use confidence as fallback
    if confidence > 0.7:
        return "High"
    elif confidence > 0.4:
        return "Medium"

    return "Low"
