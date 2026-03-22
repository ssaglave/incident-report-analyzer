"""
Student 5 — Text Analyst
Processes crime text reports using spaCy for NER,
HuggingFace for sentiment analysis and zero-shot topic classification.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Uncomment these imports as you implement each stage ──
# import pandas as pd
# import spacy
# from transformers import pipeline as hf_pipeline
# import nltk
# from nltk.corpus import stopwords
# import re


class TextPipeline(BasePipeline):
    """Text analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "text"
    SOURCE_LABEL = "text"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Text_ID", "Raw_Text",
        "Crime_Type", "Location_Entity", "Entities",
        "Sentiment", "Topic", "Severity",
    ]

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load text data from CSV or text files."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        self.raw_data = []

        # TODO: Load CSV data (CrimeReport dataset)
        # if csv_files:
        #     for csv_file in csv_files:
        #         df = pd.read_csv(csv_file)
        #         for _, row in df.iterrows():
        #             self.raw_data.append(row.to_dict())
        #
        # # Load text files
        # for txt_file in txt_files:
        #     with open(txt_file, "r", encoding="utf-8") as f:
        #         self.raw_data.append({"text": f.read(), "source_file": txt_file})

        self.logger.info(f"Found {len(csv_files)} CSV + {len(txt_files)} TXT files")
        self.logger.info(f"Loaded {len(self.raw_data)} text records")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Run NER, sentiment analysis, and topic classification."""
        self.processed_data = []

        # TODO: Implement NLP processing
        # nlp = spacy.load("en_core_web_sm")
        # sentiment_analyzer = hf_pipeline("sentiment-analysis")
        # topic_classifier = hf_pipeline("zero-shot-classification")
        # topic_labels = ["accident", "fire", "theft", "robbery", "assault", "public disturbance"]
        #
        # for item in self.raw_data:
        #     text = item.get("text", item.get("description", ""))
        #
        #     # Text preprocessing
        #     text_clean = re.sub(r'[^\w\s]', '', text)  # Remove special chars
        #     text_clean = text_clean.strip()
        #
        #     if not text_clean:
        #         continue
        #
        #     # NER
        #     doc = nlp(text_clean)
        #     entities = [(ent.text, ent.label_) for ent in doc.ents]
        #
        #     # Sentiment
        #     sentiment = sentiment_analyzer(text_clean[:512])  # Truncate for model
        #
        #     # Topic classification
        #     topic = topic_classifier(text_clean[:512], candidate_labels=topic_labels)
        #
        #     self.processed_data.append({
        #         "raw_text": text,
        #         "clean_text": text_clean,
        #         "entities": entities,
        #         "sentiment_label": sentiment[0]["label"],
        #         "sentiment_score": sentiment[0]["score"],
        #         "top_topic": topic["labels"][0],
        #         "topic_score": topic["scores"][0],
        #         "crime_type": item.get("crime_type", ""),
        #     })

        self.logger.info(f"Processed {len(self.processed_data)} text records")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from NLP analysis."""
        self.extracted_records = []

        # TODO: Implement extraction logic
        # for idx, item in enumerate(self.processed_data, start=1):
        #     # Extract locations and entities
        #     locations = [ent[0] for ent in item["entities"] if ent[1] in ("GPE", "LOC")]
        #     entity_str = ", ".join(f"{e[0]} ({e[1]})" for e in item["entities"][:5])
        #
        #     record = {
        #         "Incident_ID": generate_incident_id(idx),
        #         "Source": self.SOURCE_LABEL,
        #         "Text_ID": f"TXT_{idx:03d}",
        #         "Raw_Text": item["raw_text"][:300],  # Truncate
        #         "Crime_Type": item.get("crime_type", item["top_topic"]),
        #         "Location_Entity": ", ".join(locations) if locations else "",
        #         "Entities": entity_str,
        #         "Sentiment": item["sentiment_label"],
        #         "Topic": item["top_topic"],
        #         "Severity": classify_severity(
        #             item["raw_text"],
        #             item["topic_score"],
        #         ),
        #     }
        #     self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TextPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
