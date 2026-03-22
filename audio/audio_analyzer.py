"""
Student 1 — Audio Analyst
Processes emergency audio calls using:
  - Stage 1: Load .wav/.mp3 audio files with optional metadata CSV
  - Stage 2: Whisper (speech-to-text) + HuggingFace (sentiment)
  - Stage 3: spaCy NER (locations, persons, events) + urgency scoring

Supports THREE data modes (auto-detected):
  Mode A — Kaggle dataset (metadata CSV + linked .wav files in subfolder)
           Uses 911_metadata.csv to load metadata, Whisper to transcribe .wav
  Mode B — Standalone CSV with pre-transcribed text (skips Whisper)
  Mode C — Raw audio files only (.wav, .mp3) → Whisper transcription

Dataset: 911 Recordings: The First 6 Seconds (Kaggle)
  https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds

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


# ─── Configuration ────────────────────────────────────────────────────────────

# Limit how many audio files to process (set to None for all)
MAX_AUDIO_FILES = None

# ─── Event Classification Keywords ───────────────────────────────────────────

EVENT_KEYWORDS = {
    "Building fire / trapped persons": ["fire", "flame", "burning", "smoke", "trapped"],
    "Road accident": ["accident", "crash", "collision", "collided", "vehicle", "car", "highway"],
    "Robbery / theft": ["robbery", "robbed", "robbing", "stole", "stolen", "theft", "bank"],
    "Assault / violence": ["attack", "assault", "fight", "fighting", "knife", "stabbing", "bleeding"],
    "Public disturbance": ["noise", "loud", "party", "yelling", "complaint", "disturbance"],
    "Shooting": ["gun", "shoot", "shooting", "gunshot", "shot", "fired"],
    "Medical emergency": ["ambulance", "medical", "unconscious", "heart", "breathing", "choking"],
    "Murder / homicide": ["murder", "killed", "homicide", "dead", "death", "shot to death"],
    "Kidnapping / missing": ["kidnap", "missing", "abducted", "taken"],
    "Domestic disturbance": ["domestic", "husband", "wife", "boyfriend", "girlfriend"],
}

URGENCY_KEYWORDS = [
    "help", "hurry", "please", "emergency", "immediately", "right away",
    "trapped", "dying", "bleeding", "hurry up", "children", "gun", "weapon",
    "fire", "unconscious", "critical", "quickly", "dead", "shot", "murder",
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

    def __init__(self, data_dir: str, output_dir: str, max_files: int = None):
        super().__init__(data_dir, output_dir)

        # AI models (loaded once, reused across all files)
        self.whisper_model = None
        self.nlp = None
        self.sentiment_analyzer = None

        # Data mode: 'kaggle', 'csv', or 'audio'
        self.data_mode = None

        # Limit processing for large datasets
        self.max_files = max_files or MAX_AUDIO_FILES

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
        Load data from the data directory. Auto-detects mode:

        Mode A (Kaggle) — 911_metadata.csv found with 'filename' column:
            Loads metadata + links to .wav files for Whisper transcription.
            Dataset: kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds

        Mode B (CSV) — Other CSV with 'transcript'/'text' column:
            Loads pre-transcribed text directly, skips Whisper.

        Mode C (Audio) — Only .wav/.mp3 files, no CSV:
            Whisper transcribes each audio file.
        """
        self.raw_data = []

        # ── Check for Kaggle metadata CSV (Mode A) ──
        kaggle_csv = self._find_kaggle_metadata()
        if kaggle_csv:
            self._load_kaggle_data(kaggle_csv)
            return

        # ── Check for other CSV files (Mode B) ──
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if csv_files:
            self._load_csv_data(csv_files)
            return

        # ── Mode C: Raw audio files ──
        self._load_audio_files()

    def _find_kaggle_metadata(self) -> str:
        """Search for the Kaggle 911_metadata.csv in data_dir and its subfolders."""
        # Check data_dir directly
        direct = os.path.join(self.data_dir, "911_metadata.csv")
        if os.path.exists(direct):
            return direct

        # Check subfolders (Kaggle extracts into 911_first6sec/)
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f == "911_metadata.csv":
                    return os.path.join(root, f)

        return None

    def _load_kaggle_data(self, csv_path: str):
        """
        Mode A: Load Kaggle 911 Recordings dataset.
        The CSV has metadata (title, date, state, deaths, description)
        and a 'filename' column pointing to .wav files.
        """
        self.data_mode = "kaggle"
        self.logger.info(f"Mode A (Kaggle): Loading {csv_path}")

        df = pd.read_csv(csv_path)
        self.logger.info(f"  Columns: {list(df.columns)}")
        self.logger.info(f"  Total rows: {len(df)}")

        csv_dir = os.path.dirname(csv_path)
        # The 'filename' column has paths like '911_first6sec/call_2_0.wav'
        # We need to resolve them relative to csv_dir's parent (data_dir)

        for _, row in df.iterrows():
            filename = str(row.get("filename", ""))
            if not filename or filename == "nan":
                continue

            # Resolve audio file path
            # filename could be "911_first6sec/call_2_0.wav" (relative to data_dir)
            audio_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(audio_path):
                # Try relative to the CSV file's directory
                audio_path = os.path.join(csv_dir, os.path.basename(filename))
            if not os.path.exists(audio_path):
                # Try just the basename in csv_dir
                audio_path = os.path.join(csv_dir, filename.split("/")[-1])

            self.raw_data.append({
                "audio_file": audio_path if os.path.exists(audio_path) else None,
                "filename": filename,
                "title": str(row.get("title", "")),
                "date": str(row.get("date", "")),
                "state": str(row.get("state", "")),
                "deaths": str(row.get("deaths", "")),
                "potential_death": str(row.get("potential_death", "")),
                "false_alarm": str(row.get("false_alarm", "")),
                "description": str(row.get("description", "")),
                "event_id": str(row.get("event_id", "")),
            })

        # Apply limit if set
        if self.max_files and len(self.raw_data) > self.max_files:
            self.logger.info(
                f"  Limiting to {self.max_files} files (of {len(self.raw_data)} total)"
            )
            self.raw_data = self.raw_data[: self.max_files]

        # Count how many have valid audio files
        valid_audio = sum(1 for r in self.raw_data if r["audio_file"])
        self.logger.info(
            f"  Loaded {len(self.raw_data)} records "
            f"({valid_audio} with valid audio files)"
        )

    def _load_csv_data(self, csv_files: list):
        """Mode B: Load pre-transcribed CSV data."""
        self.data_mode = "csv"

        for csv_file in csv_files:
            self.logger.info(f"Mode B (CSV): Loading {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file)
            self.logger.info(f"  Columns: {list(df.columns)}")

            transcript_col = self._find_column(df, [
                "transcript", "transcription", "text",
                "Transcript", "Transcription", "Text",
            ])
            date_col = self._find_column(df, ["date", "Date", "timestamp"])
            state_col = self._find_column(df, [
                "state", "State", "location", "Location",
            ])
            deaths_col = self._find_column(df, ["deaths", "Deaths", "fatalities"])

            for _, row in df.iterrows():
                transcript = str(row.get(transcript_col, "")) if transcript_col else ""
                if not transcript.strip() or transcript == "nan":
                    continue
                self.raw_data.append({
                    "transcript": transcript.strip(),
                    "date": str(row.get(date_col, "")) if date_col else "",
                    "state": str(row.get(state_col, "")) if state_col else "",
                    "deaths": str(row.get(deaths_col, "")) if deaths_col else "",
                    "description": "",
                })

        if self.max_files and len(self.raw_data) > self.max_files:
            self.raw_data = self.raw_data[: self.max_files]

        self.logger.info(f"  Loaded {len(self.raw_data)} pre-transcribed records")

    def _load_audio_files(self):
        """Mode C: Load raw audio files for Whisper transcription."""
        self.data_mode = "audio"
        audio_extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")

        files = []
        for ext in audio_extensions:
            files.extend(sorted(glob.glob(os.path.join(self.data_dir, ext))))

        if self.max_files and len(files) > self.max_files:
            files = files[: self.max_files]

        self.raw_data = [{"audio_file": f, "filename": os.path.basename(f)} for f in files]

        if not self.raw_data:
            self.logger.warning(
                f"No data found in {self.data_dir}.\n"
                "  Option 1: python audio/generate_samples.py (test data)\n"
                "  Option 2: kaggle datasets download -d louisteitelbaum/"
                "911-recordings-first-6-seconds -p audio/data/ --unzip"
            )

        self.logger.info(f"Mode C (Audio): Found {len(self.raw_data)} audio files")

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
        - Mode A (Kaggle): Whisper on .wav files + sentiment on (transcript + description)
        - Mode B (CSV): Sentiment on pre-transcribed text only
        - Mode C (Audio): Whisper + sentiment
        """
        need_whisper = self.data_mode in ("kaggle", "audio")
        if self.nlp is None:
            self._load_models(need_whisper=need_whisper)

        self.processed_data = []

        if self.data_mode == "kaggle":
            self._process_kaggle_data()
        elif self.data_mode == "csv":
            self._process_csv_data()
        else:
            self._process_audio_data()

        self.logger.info(f"Processed {len(self.processed_data)} records")

    def _process_kaggle_data(self):
        """Mode A: Transcribe Kaggle .wav files + combine with metadata description."""
        total = len(self.raw_data)

        for i, item in enumerate(self.raw_data, start=1):
            # ── Whisper: Transcribe the .wav file ──
            whisper_text = ""
            audio_file = item.get("audio_file")

            if audio_file and os.path.exists(audio_file):
                self.logger.info(f"[{i}/{total}] Transcribing: {item['filename']}")
                try:
                    result = self.whisper_model.transcribe(audio_file)
                    whisper_text = result["text"].strip()
                except Exception as e:
                    self.logger.error(f"  Whisper error: {e}")
            else:
                self.logger.info(f"[{i}/{total}] No audio file, using description only")

            # ── Combine Whisper transcript + metadata description ──
            description = item.get("description", "")
            if description == "nan":
                description = ""

            # Use both sources: Whisper transcription + event description
            combined_text = ""
            if whisper_text and description:
                combined_text = f"{whisper_text} | Context: {description}"
            elif whisper_text:
                combined_text = whisper_text
            elif description:
                combined_text = description

            if not combined_text.strip():
                self.logger.warning(f"  No text for record {i}, skipping.")
                continue

            # ── Sentiment Analysis ──
            sentiment_display, sentiment_score = self._analyze_sentiment(combined_text)

            self.processed_data.append({
                "filename": item["filename"],
                "whisper_transcript": whisper_text,
                "description": description,
                "combined_text": combined_text,
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "state": item.get("state", ""),
                "deaths": item.get("deaths", ""),
                "potential_death": item.get("potential_death", ""),
                "false_alarm": item.get("false_alarm", ""),
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
            })

            self.logger.info(
                f"  ✓ Whisper: \"{whisper_text[:60]}...\" "
                f"| Sentiment: {sentiment_display} ({sentiment_score:.2f})"
            )

    def _process_csv_data(self):
        """Mode B: Run sentiment analysis on pre-transcribed text."""
        for i, item in enumerate(self.raw_data, start=1):
            transcript = item["transcript"]
            self.logger.info(f"[{i}/{len(self.raw_data)}] Analyzing: {transcript[:60]}...")

            sentiment_display, sentiment_score = self._analyze_sentiment(transcript)

            self.processed_data.append({
                "filename": "",
                "whisper_transcript": "",
                "description": item.get("description", ""),
                "combined_text": transcript,
                "title": "",
                "date": item.get("date", ""),
                "state": item.get("state", ""),
                "deaths": item.get("deaths", ""),
                "potential_death": "",
                "false_alarm": "",
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
            })

    def _process_audio_data(self):
        """Mode C: Transcribe audio files with Whisper + sentiment."""
        for i, item in enumerate(self.raw_data, start=1):
            audio_file = item["audio_file"]
            filename = item["filename"]
            self.logger.info(f"[{i}/{len(self.raw_data)}] Transcribing: {filename}")

            try:
                result = self.whisper_model.transcribe(audio_file)
                transcript = result["text"].strip()
            except Exception as e:
                self.logger.error(f"Whisper error on {filename}: {e}")
                transcript = ""

            if not transcript:
                self.logger.warning(f"Empty transcript for {filename}, skipping.")
                continue

            sentiment_display, sentiment_score = self._analyze_sentiment(transcript)

            self.processed_data.append({
                "filename": filename,
                "whisper_transcript": transcript,
                "description": "",
                "combined_text": transcript,
                "title": "",
                "date": "",
                "state": "",
                "deaths": "",
                "potential_death": "",
                "false_alarm": "",
                "sentiment_label": sentiment_display,
                "sentiment_score": sentiment_score,
            })

            self.logger.info(
                f"  ✓ Transcript: {transcript[:80]}..."
                f" | Sentiment: {sentiment_display} ({sentiment_score:.2f})"
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
        Extract structured fields from processed data:
        - spaCy NER for locations, persons
        - Keyword matching for event classification
        - Urgency scoring based on keywords + sentiment + metadata
        - Use Kaggle metadata (date, state, deaths) when available
        """
        self.extracted_records = []

        for idx, item in enumerate(self.processed_data, start=1):
            text = item["combined_text"]

            # ── spaCy NER: Extract entities ──
            doc = self.nlp(text[:5000])  # Limit for performance

            locations = []
            persons = []
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    locations.append(ent.text)
                elif ent.label_ == "PERSON":
                    persons.append(ent.text)

            # Use state from metadata as fallback location
            state = item.get("state", "")
            if state and state != "nan" and state not in locations:
                locations.insert(0, state)

            # ── Event Classification ──
            extracted_event = self._classify_event(text)

            # Use title from Kaggle metadata as supplementary event info
            title = item.get("title", "")
            if title and title != "nan":
                title_event = self._classify_event(title)
                if title_event != "Unknown incident":
                    extracted_event = title_event

            # ── Urgency Score ──
            urgency_score = self._calculate_urgency(
                text, item["sentiment_label"], item["sentiment_score"]
            )

            # Boost urgency for deaths / potential death
            deaths = item.get("deaths", "")
            if deaths and deaths not in ("", "0", "0.0", "nan"):
                urgency_score = min(urgency_score + 0.2, 1.0)
            potential_death = item.get("potential_death", "")
            if potential_death and potential_death not in ("", "0", "0.0", "nan"):
                urgency_score = min(urgency_score + 0.1, 1.0)

            # ── Severity ──
            severity = classify_severity(text, urgency_score)

            # ── Timestamp ──
            timestamp = item.get("date", "")
            if timestamp == "nan":
                timestamp = ""

            # ── Transcript (show Whisper output or combined text) ──
            transcript = item.get("whisper_transcript", "") or text

            # ── Build record following common schema ──
            record = {
                "Incident_ID": generate_incident_id(idx),
                "Source": self.SOURCE_LABEL,
                "Call_ID": f"C{idx:03d}",
                "Transcript": transcript,
                "Extracted_Event": extracted_event,
                "Location": ", ".join(dict.fromkeys(locations)) if locations else "",
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
        keyword_score = min(keyword_hits / 5.0, 1.0)

        # Sentiment-based urgency (distressed = high urgency)
        if sentiment == "Distressed":
            sentiment_urgency = sentiment_score
        else:
            sentiment_urgency = 1.0 - sentiment_score

        # Weighted combination
        urgency = (0.6 * keyword_score) + (0.4 * sentiment_urgency)
        return min(urgency, 1.0)


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio analysis pipeline")
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max number of audio files to process (default: all)",
    )
    args = parser.parse_args()

    root = os.path.dirname(__file__)
    pipeline = AudioPipeline(
        data_dir=os.path.join(root, "data"),
        output_dir=os.path.join(root, "output"),
        max_files=args.max,
    )
    results = pipeline.run()

    # Display summary
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
