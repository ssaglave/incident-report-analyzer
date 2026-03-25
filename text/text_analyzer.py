"""
Student 5 — Text Analyst
Processes crime text reports using spaCy for NER,
HuggingFace for sentiment analysis and zero-shot topic classification.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob
import json
import re
from typing import List, Tuple

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity


class TextPipeline(BasePipeline):
    """Text analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "text"
    SOURCE_LABEL = "text"
    OUTPUT_COLUMNS = [
        "Incident_ID",
        "Text_ID",
        "Crime_Type",
        "Location_Entity",
        "Sentiment",
        "Topic",
        "Severity",
    ]

    def __init__(self, data_dir, output_dir):
        super().__init__(data_dir, output_dir)
        self.nlp = None
        self.sentiment_analyzer = None
        self.topic_classifier = None
        self.topic_labels = [
            "shooting",
            "robbery",
            "theft",
            "fire",
            "assault",
            "traffic incident",
            "police investigation",
            "public disturbance",
        ]

    def _clean_text(self, text: str) -> str:
        """Clean social media text."""
        if not isinstance(text, str):
            return ""

        text = text.strip()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _format_entities(self, entities: List[Tuple[str, str]]) -> str:
        """Format entity list for CSV output."""
        if not entities:
            return ""

        formatted = []
        seen = set()

        for ent_text, ent_label in entities:
            key = (ent_text.strip(), ent_label)
            if key not in seen:
                seen.add(key)
                formatted.append(f"{ent_text} ({ent_label})")

        return ", ".join(formatted[:8])

    def _find_text_column(self, df: pd.DataFrame) -> str:
        """Find likely text column in CSV if any CSV is used."""
        candidate_columns = [
            "text",
            "description",
            "details",
            "report",
            "content",
            "summary",
            "incident_text",
        ]

        lower_map = {col.lower(): col for col in df.columns}
        for candidate in candidate_columns:
            if candidate in lower_map:
                return lower_map[candidate]

        for col in df.columns:
            if df[col].dtype == "object":
                return col

        raise ValueError("Could not find a usable text column in the CSV.")

    def _find_optional_column(self, df: pd.DataFrame, candidates: List[str]) -> str:
        """Find optional columns like location or crime type."""
        lower_map = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate in lower_map:
                return lower_map[candidate]
        return ""

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load text data from CSV or JSON-lines TXT files."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        self.raw_data = []

        # Optional CSV loading support
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            text_col = self._find_text_column(df)
            crime_col = self._find_optional_column(
                df, ["crime_type", "category", "label", "type", "incident_type"]
            )
            location_col = self._find_optional_column(
                df, ["location", "place", "address", "area", "street"]
            )

            for _, row in df.iterrows():
                self.raw_data.append(
                    {
                        "text": row.get(text_col, ""),
                        "crime_type": row.get(crime_col, "") if crime_col else "",
                        "location": row.get(location_col, "") if location_col else "",
                        "source_file": os.path.basename(csv_file),
                        "created_at": "",
                        "tweet_id": "",
                        "is_retweet": False,
                    }
                )

        # TXT is JSON-lines Twitter data
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    tweet_text = record.get("text", "")
                    created_at = record.get("created_at", "")
                    tweet_id = record.get("id", "")

                    place_name = ""
                    place = record.get("place")
                    if isinstance(place, dict):
                        place_name = place.get("full_name", "")

                    is_retweet = "retweeted_status" in record or str(tweet_text).startswith("RT @")

                    self.raw_data.append(
                        {
                            "text": tweet_text,
                            "crime_type": "",
                            "location": place_name,
                            "source_file": os.path.basename(txt_file),
                            "created_at": created_at,
                            "tweet_id": tweet_id,
                            "is_retweet": is_retweet,
                        }
                    )

        self.logger.info(f"Found {len(csv_files)} CSV + {len(txt_files)} TXT files")
        self.logger.info(f"Loaded {len(self.raw_data)} text records")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Run NER, sentiment analysis, and topic classification."""
        self.processed_data = []

        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = hf_pipeline("sentiment-analysis")
        self.topic_classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

        for item in self.raw_data:
            raw_text = str(item.get("text", ""))
            text_clean = self._clean_text(raw_text)

            if not text_clean or len(text_clean.split()) < 4:
                continue

            if item.get("is_retweet", False):
                continue

            # NER
            doc = self.nlp(text_clean)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Sentiment
            sentiment = self.sentiment_analyzer(text_clean[:512])

            # Topic classification
            topic = self.topic_classifier(
                text_clean[:512],
                candidate_labels=self.topic_labels,
            )

            self.processed_data.append(
                {
                    "raw_text": raw_text,
                    "clean_text": text_clean,
                    "entities": entities,
                    "sentiment_label": sentiment[0]["label"],
                    "sentiment_score": float(sentiment[0]["score"]),
                    "top_topic": topic["labels"][0],
                    "topic_score": float(topic["scores"][0]),
                    "crime_type": item.get("crime_type", ""),
                    "location": item.get("location", ""),
                    "source_file": item.get("source_file", ""),
                    "created_at": item.get("created_at", ""),
                    "tweet_id": item.get("tweet_id", ""),
                }
            )

        self.logger.info(f"Processed {len(self.processed_data)} text records")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from NLP analysis."""
        self.extracted_records = []

        for idx, item in enumerate(self.processed_data, start=1):
            locations = [
                ent_text
                for ent_text, ent_label in item["entities"]
                if ent_label in ("GPE", "LOC", "FAC")
            ]

            if not locations and item.get("location"):
                locations = [str(item["location"])]

            crime_type = item["crime_type"] if item["crime_type"] else item["top_topic"]

            record = {
                "Incident_ID": generate_incident_id(idx),
                "Text_ID": f"TXT_{idx:03d}",
                "Crime_Type": crime_type,
                "Location_Entity": ", ".join(dict.fromkeys(locations)) if locations else "",
                "Sentiment": item["sentiment_label"],
                "Topic": item["top_topic"],
                "Severity": classify_severity(
                    item["raw_text"],
                    item["topic_score"],
                ),
            }

            self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TextPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results.head())