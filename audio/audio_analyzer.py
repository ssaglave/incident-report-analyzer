"""
Student 1 — Audio Analyst
Processes emergency audio calls using Whisper for speech-to-text,
spaCy for NER, and HuggingFace for sentiment analysis.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob

# Add project root to path so we can import the common pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Uncomment these imports as you implement each stage ──
# import whisper
# import spacy
# from transformers import pipeline as hf_pipeline


class AudioPipeline(BasePipeline):
    """Audio analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "audio"
    SOURCE_LABEL = "audio"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Call_ID", "Transcript",
        "Extracted_Event", "Location", "Timestamp",
        "Sentiment", "Urgency_Score", "Severity",
    ]

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load audio files (.wav, .mp3) from data directory."""
        audio_extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")
        self.raw_data = []

        for ext in audio_extensions:
            self.raw_data.extend(glob.glob(os.path.join(self.data_dir, ext)))

        self.logger.info(f"Found {len(self.raw_data)} audio files")

        # TODO: If using Kaggle notebook output, load the pre-transcribed text
        # self.raw_data = pd.read_csv(os.path.join(self.data_dir, "transcripts.csv"))

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Run speech-to-text and NLP models on audio data."""
        self.processed_data = []

        # TODO: Implement Whisper transcription
        # model = whisper.load_model("base")
        # for audio_file in self.raw_data:
        #     result = model.transcribe(audio_file)
        #     self.processed_data.append({
        #         "file": audio_file,
        #         "transcript": result["text"],
        #     })

        # TODO: Run sentiment analysis on transcripts
        # sentiment_analyzer = hf_pipeline("sentiment-analysis")
        # for item in self.processed_data:
        #     sentiment = sentiment_analyzer(item["transcript"])
        #     item["sentiment"] = sentiment[0]["label"]
        #     item["sentiment_score"] = sentiment[0]["score"]

        self.logger.info(f"Processed {len(self.processed_data)} audio files")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from processed audio data."""
        self.extracted_records = []

        # TODO: Implement extraction logic
        # nlp = spacy.load("en_core_web_sm")
        #
        # for idx, item in enumerate(self.processed_data, start=1):
        #     doc = nlp(item["transcript"])
        #
        #     # Extract locations from NER
        #     locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
        #
        #     # Build record following the common schema
        #     record = {
        #         "Incident_ID": generate_incident_id(idx),
        #         "Source": self.SOURCE_LABEL,
        #         "Call_ID": f"C{idx:03d}",
        #         "Transcript": item["transcript"][:500],  # Truncate
        #         "Extracted_Event": "TODO: classify event type",
        #         "Location": ", ".join(locations) if locations else "",
        #         "Timestamp": "",
        #         "Sentiment": item.get("sentiment", ""),
        #         "Urgency_Score": item.get("sentiment_score", 0.0),
        #         "Severity": classify_severity(item["transcript"], item.get("sentiment_score", 0)),
        #     }
        #     self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = AudioPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
