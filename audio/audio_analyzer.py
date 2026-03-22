"""
Student 1 — Audio Analyst
Processes emergency audio calls using:
  - Stage 1: Load .wav/.mp3 audio files OR pre-transcribed CSV from Kaggle
  - Stage 2: Whisper (speech-to-text) + HuggingFace (sentiment)
  - Stage 3: spaCy NER (locations, persons, events) + urgency scoring

Supports TWO data modes:
  Mode A — Raw audio files (.wav, .mp3) → Whisper transcription → NLP
  Mode B — Kaggle CSV with pre-transcribed text → NLP only (skips Whisper)

Dataset: 911 Recordings: The First 6 Seconds (Kaggle)
  https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2
  https://www.kaggle.com/datasets/louisedavis/911-recordings-the-first-6-seconds

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob
import re

# Add project root to path so we can import the common pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

import pandas as pd
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

        # Data mode: 'audio' (raw files) or 'csv' (pre-transcribed)
        self.data_mode = None

    def _load_models(self, need_whisper: bool = True):
        """Load all AI models before processing."""
        if need_whisper:
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
        """
        Load data from the data directory. Supports two modes:

        Mode A — Raw audio files (.wav, .mp3, .flac):
            Whisper will transcribe each file in Stage 2.

        Mode B — Pre-transcribed CSV (from Kaggle notebook output):
            If a CSV file is found in data_dir, it loads transcripts directly.
            This is useful when using the Kaggle 911 Calls + Wav2Vec2 notebook
            which already produces transcriptions.

            Expected CSV columns (flexible — will auto-detect):
              - transcript / text / transcription  (the transcribed text)
              - date / Date                        (optional)
              - state / State / location           (optional)
              - deaths / Deaths                    (optional)
        """
        # ── Check for CSV files first (Mode B: pre-transcribed) ──
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))

        if csv_files:
            self.data_mode = "csv"
            self.raw_data = []

            for csv_file in csv_files:
                self.logger.info(f"Loading pre-transcribed CSV: {os.path.basename(csv_file)}")
                df = pd.read_csv(csv_file)
                self.logger.info(f"  Columns: {list(df.columns)}")
                self.logger.info(f"  Rows: {len(df)}")

                # Auto-detect the transcript column
                transcript_col = self._find_column(df, [
                    "transcript", "transcription", "text", "description",
                    "Transcript", "Transcription", "Text", "Description",
                ])

                # Auto-detect optional metadata columns
                date_col = self._find_column(df, ["date", "Date", "DATE", "timestamp"])
                state_col = self._find_column(df, [
                    "state", "State", "location", "Location", "city", "City",
                ])
                deaths_col = self._find_column(df, ["deaths", "Deaths", "fatalities"])

                for _, row in df.iterrows():
                    transcript = str(row.get(transcript_col, "")) if transcript_col else ""
                    if not transcript.strip() or transcript == "nan":
                        continue

                    self.raw_data.append({
                        "source_file": csv_file,
                        "transcript": transcript.strip(),
                        "date": str(row.get(date_col, "")) if date_col else "",
                        "location": str(row.get(state_col, "")) if state_col else "",
                        "deaths": str(row.get(deaths_col, "")) if deaths_col else "",
                    })

            self.logger.info(
                f"Mode B (CSV): Loaded {len(self.raw_data)} pre-transcribed records "
                f"from {len(csv_files)} file(s)"
            )
            return

        # ── Mode A: Raw audio files ──
        audio_extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")
        self.raw_data = []
        self.data_mode = "audio"

        for ext in audio_extensions:
            self.raw_data.extend(
                sorted(glob.glob(os.path.join(self.data_dir, ext)))
            )

        if not self.raw_data:
            self.logger.warning(
                f"No audio or CSV files found in {self.data_dir}.\n"
                "  Option 1: Run 'python audio/generate_samples.py' for test data\n"
                "  Option 2: Download Kaggle dataset (see audio/README.md)"
            )

        self.logger.info(f"Mode A (Audio): Found {len(self.raw_data)} audio files")

    def _find_column(self, df: pd.DataFrame, candidates: list) -> str:
        """Find the first matching column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2: AI PROCESSING
    # ═════════════════════════════════════════════════════════════════════════

    def process_data(self):
        """
        Run AI models on the data.
        - Mode A (audio): Whisper transcription + sentiment analysis
        - Mode B (CSV): Sentiment analysis only (transcripts already available)
        """
        # Load models based on mode
        need_whisper = (self.data_mode == "audio")
        if self.nlp is None:
            self._load_models(need_whisper=need_whisper)

        self.processed_data = []

        if self.data_mode == "csv":
            self._process_csv_data()
        else:
            self._process_audio_data()

        self.logger.info(f"Processed {len(self.processed_data)} records")

    def _process_audio_data(self):
        """Mode A: Transcribe audio files with Whisper + sentiment analysis."""
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

            # ── Sentiment Analysis ──
            sentiment_display, sentiment_score = self._analyze_sentiment(transcript)

            self.processed_data.append({
                "file": audio_file,
                "filename": filename,
                "transcript": transcript,
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
                "date": "",
                "csv_location": "",
                "deaths": "",
            })

            self.logger.info(
                f"  ✓ Transcript: {transcript[:80]}..."
                f" | Sentiment: {sentiment_display} ({sentiment_score:.2f})"
            )

    def _process_csv_data(self):
        """Mode B: Run sentiment analysis on pre-transcribed text from CSV."""
        for i, item in enumerate(self.raw_data, start=1):
            transcript = item["transcript"]
            self.logger.info(f"[{i}/{len(self.raw_data)}] Analyzing: {transcript[:60]}...")

            # ── Sentiment Analysis ──
            sentiment_display, sentiment_score = self._analyze_sentiment(transcript)

            self.processed_data.append({
                "file": item["source_file"],
                "filename": os.path.basename(item["source_file"]),
                "transcript": transcript,
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
                "date": item.get("date", ""),
                "csv_location": item.get("location", ""),
                "deaths": item.get("deaths", ""),
            })

            self.logger.info(
                f"  ✓ Sentiment: {sentiment_display} ({sentiment_score:.2f})"
            )

    def _analyze_sentiment(self, text: str) -> tuple:
        """Run HuggingFace sentiment analysis. Returns (label, score)."""
        try:
            result = self.sentiment_analyzer(text[:512])
            label = result[0]["label"]
            score = result[0]["score"]
        except Exception as e:
            self.logger.error(f"Sentiment error: {e}")
            label = "UNKNOWN"
            score = 0.5

        # Map to human-readable labels
        if label == "NEGATIVE":
            display = "Distressed"
        elif label == "POSITIVE":
            display = "Calm"
        else:
            display = label

        return display, score

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3: INFORMATION EXTRACTION
    # ═════════════════════════════════════════════════════════════════════════

    def extract_information(self):
        """
        Extract structured fields from transcripts:
        - spaCy NER for locations, persons
        - Keyword matching for event classification
        - Urgency scoring based on keywords + sentiment
        - Use CSV metadata (date, state) when available
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

            # If CSV provided a location/state, use it as fallback
            csv_location = item.get("csv_location", "")
            if csv_location and csv_location != "nan" and not locations:
                locations.append(csv_location)

            # ── Event Classification ──
            extracted_event = self._classify_event(transcript)

            # ── Urgency Score ──
            urgency_score = self._calculate_urgency(
                transcript, item["sentiment_label"], item["sentiment_score"]
            )

            # Boost urgency if deaths are reported in CSV metadata
            deaths = item.get("deaths", "")
            if deaths and deaths not in ("", "0", "nan"):
                urgency_score = min(urgency_score + 0.2, 1.0)

            # ── Severity ──
            severity = classify_severity(transcript, urgency_score)

            # ── Timestamp from CSV metadata ──
            timestamp = item.get("date", "")
            if timestamp == "nan":
                timestamp = ""

            # ── Build record following common schema ──
            record = {
                "Incident_ID": generate_incident_id(idx),
                "Source": self.SOURCE_LABEL,
                "Call_ID": f"C{idx:03d}",
                "Transcript": transcript,
                "Extracted_Event": extracted_event,
                "Location": ", ".join(locations) if locations else "",
                "Timestamp": timestamp,
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
