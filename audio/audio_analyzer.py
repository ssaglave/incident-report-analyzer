"""
Student 1 — Audio Analyst
Processes emergency audio calls using:
  - Stage 1: Load .wav/.mp3 audio files
  - Stage 2: Whisper (speech-to-text) + HuggingFace (sentiment)
  - Stage 3: spaCy NER (locations, persons, events) + urgency scoring

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob
import re

# Add project root to path so we can import the common pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

import whisper
import spacy
from transformers import pipeline as hf_pipeline


# ─── Event Classification Keywords ───────────────────────────────────────────

EVENT_KEYWORDS = {
    "Building fire / trapped persons": ["fire", "flame", "burning", "smoke", "trapped"],
    "Road accident": ["accident", "crash", "collision", "collided", "vehicle", "car", "highway"],
    "Robbery / theft": ["robbery", "robbed", "robbing", "stole", "stolen", "theft", "bank"],
    "Assault / violence": ["attack", "assault", "fight", "fighting", "knife", "stabbing", "bleeding"],
    "Public disturbance": ["noise", "loud", "party", "yelling", "complaint", "disturbance"],
    "Shooting": ["gun", "shoot", "shooting", "gunshot", "shot"],
    "Medical emergency": ["ambulance", "medical", "unconscious", "heart", "breathing", "choking"],
}

URGENCY_KEYWORDS = [
    "help", "hurry", "please", "emergency", "immediately", "right away",
    "trapped", "dying", "bleeding", "hurry up", "children", "gun", "weapon",
    "fire", "unconscious", "critical", "quickly",
]


class AudioPipeline(BasePipeline):
    """Audio analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "audio"
    SOURCE_LABEL = "audio"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Call_ID", "Transcript",
        "Extracted_Event", "Location", "Timestamp",
        "Sentiment", "Urgency_Score", "Severity",
    ]

    def __init__(self, data_dir: str, output_dir: str):
        super().__init__(data_dir, output_dir)

        # AI models (loaded once, reused across all files)
        self.whisper_model = None
        self.nlp = None
        self.sentiment_analyzer = None

    def _load_models(self):
        """Load all AI models before processing."""
        self.logger.info("Loading Whisper model (base)...")
        self.whisper_model = whisper.load_model("base")

        self.logger.info("Loading spaCy NER model...")
        self.nlp = spacy.load("en_core_web_sm")

        self.logger.info("Loading HuggingFace sentiment analyzer...")
        self.sentiment_analyzer = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        self.logger.info("All models loaded ✓")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1: DATA INGESTION
    # ═════════════════════════════════════════════════════════════════════════

    def load_data(self):
        """Load audio files (.wav, .mp3, .flac) from data directory."""
        audio_extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")
        self.raw_data = []

        for ext in audio_extensions:
            self.raw_data.extend(
                sorted(glob.glob(os.path.join(self.data_dir, ext)))
            )

        if not self.raw_data:
            self.logger.warning(
                f"No audio files found in {self.data_dir}. "
                "Run 'python audio/generate_samples.py' to create test data."
            )

        self.logger.info(f"Found {len(self.raw_data)} audio files")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2: AI PROCESSING
    # ═════════════════════════════════════════════════════════════════════════

    def process_data(self):
        """
        Run Whisper speech-to-text and HuggingFace sentiment analysis
        on each audio file.
        """
        # Load models on first run
        if self.whisper_model is None:
            self._load_models()

        self.processed_data = []

        for i, audio_file in enumerate(self.raw_data, start=1):
            filename = os.path.basename(audio_file)
            self.logger.info(f"[{i}/{len(self.raw_data)}] Transcribing: {filename}")

            # ── Whisper: Audio → Text ──
            try:
                result = self.whisper_model.transcribe(audio_file)
                transcript = result["text"].strip()
            except Exception as e:
                self.logger.error(f"Whisper error on {filename}: {e}")
                transcript = ""

            if not transcript:
                self.logger.warning(f"Empty transcript for {filename}, skipping.")
                continue

            # ── HuggingFace: Sentiment Analysis ──
            try:
                # Truncate to 512 tokens for the model
                sentiment_result = self.sentiment_analyzer(transcript[:512])
                sentiment_label = sentiment_result[0]["label"]   # POSITIVE / NEGATIVE
                sentiment_score = sentiment_result[0]["score"]   # 0.0–1.0
            except Exception as e:
                self.logger.error(f"Sentiment error on {filename}: {e}")
                sentiment_label = "UNKNOWN"
                sentiment_score = 0.5

            # Map sentiment to human-readable labels
            if sentiment_label == "NEGATIVE":
                sentiment_display = "Distressed"
            elif sentiment_label == "POSITIVE":
                sentiment_display = "Calm"
            else:
                sentiment_display = sentiment_label

            self.processed_data.append({
                "file": audio_file,
                "filename": filename,
                "transcript": transcript,
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
            })

            self.logger.info(
                f"  ✓ Transcript: {transcript[:80]}..."
                f" | Sentiment: {sentiment_display} ({sentiment_score:.2f})"
            )

        self.logger.info(f"Processed {len(self.processed_data)} audio files")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3: INFORMATION EXTRACTION
    # ═════════════════════════════════════════════════════════════════════════

    def extract_information(self):
        """
        Extract structured fields from transcripts:
        - spaCy NER for locations, persons
        - Keyword matching for event classification
        - Urgency scoring based on keywords + sentiment
        """
        self.extracted_records = []

        for idx, item in enumerate(self.processed_data, start=1):
            transcript = item["transcript"]

            # ── spaCy NER: Extract entities ──
            doc = self.nlp(transcript)

            locations = []
            persons = []
            orgs = []
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    locations.append(ent.text)
                elif ent.label_ == "PERSON":
                    persons.append(ent.text)
                elif ent.label_ == "ORG":
                    orgs.append(ent.text)

            # ── Event Classification ──
            extracted_event = self._classify_event(transcript)

            # ── Urgency Score ──
            urgency_score = self._calculate_urgency(
                transcript, item["sentiment_label"], item["sentiment_score"]
            )

            # ── Severity ──
            severity = classify_severity(transcript, urgency_score)

            # ── Build record following common schema ──
            record = {
                "Incident_ID": generate_incident_id(idx),
                "Source": self.SOURCE_LABEL,
                "Call_ID": f"C{idx:03d}",
                "Transcript": transcript,
                "Extracted_Event": extracted_event,
                "Location": ", ".join(locations) if locations else "",
                "Timestamp": "",  # No timestamp in audio; can be filled during integration
                "Sentiment": item["sentiment_label"],
                "Urgency_Score": round(urgency_score, 2),
                "Severity": severity,
            }
            self.extracted_records.append(record)

            self.logger.info(
                f"  Record {idx}: Event={extracted_event} | "
                f"Location={record['Location'] or 'N/A'} | "
                f"Urgency={urgency_score:.2f} | Severity={severity}"
            )

        self.logger.info(f"Extracted {len(self.extracted_records)} records")

    # ── Helper Methods ──────────────────────────────────────────────────────

    def _classify_event(self, transcript: str) -> str:
        """Classify the incident type based on keyword matching."""
        text_lower = transcript.lower()
        best_event = "Unknown incident"
        best_score = 0

        for event_type, keywords in EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_event = event_type

        return best_event

    def _calculate_urgency(
        self, transcript: str, sentiment: str, sentiment_score: float
    ) -> float:
        """
        Calculate urgency score (0.0–1.0) based on:
        - Presence of urgency keywords (60% weight)
        - Sentiment analysis result (40% weight)
        """
        text_lower = transcript.lower()

        # Keyword-based urgency
        keyword_hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
        keyword_score = min(keyword_hits / 5.0, 1.0)  # Cap at 1.0

        # Sentiment-based urgency (distressed = high urgency)
        if sentiment == "Distressed":
            sentiment_urgency = sentiment_score
        else:
            sentiment_urgency = 1.0 - sentiment_score  # Calm = low urgency

        # Weighted combination
        urgency = (0.6 * keyword_score) + (0.4 * sentiment_urgency)
        return min(urgency, 1.0)


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__))

    pipeline = AudioPipeline(
        data_dir=os.path.join(root, "data"),
        output_dir=os.path.join(root, "output"),
    )
    results = pipeline.run()

    # Display results
    if not results.empty:
        print("\n" + "=" * 80)
        print("  AUDIO ANALYSIS RESULTS")
        print("=" * 80)
        for _, row in results.iterrows():
            print(f"\n  {row['Call_ID']} ({row['Incident_ID']}):")
            print(f"    Event:    {row['Extracted_Event']}")
            print(f"    Location: {row['Location'] or '—'}")
            print(f"    Urgency:  {row['Urgency_Score']} | Sentiment: {row['Sentiment']}")
            print(f"    Severity: {row['Severity']}")
            print(f"    Transcript: {row['Transcript'][:100]}...")
