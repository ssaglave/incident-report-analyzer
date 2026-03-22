"""
Student 2 — Document Analyst
Processes police report PDFs using pdfplumber for text extraction,
pytesseract for OCR on scanned docs, and spaCy for NER.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Uncomment these imports as you implement each stage ──
# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import spacy


class PDFPipeline(BasePipeline):
    """PDF analysis pipeline — plugs into the common 5-stage framework."""

    MODULE_NAME = "pdf"
    SOURCE_LABEL = "pdf"
    OUTPUT_COLUMNS = [
        "Incident_ID", "Source", "Report_ID", "Department",
        "Doc_Type", "Date", "Location", "Officer", "Summary", "Severity",
    ]

    # ── STAGE 1: Data Ingestion ─────────────────────────────────────────────

    def load_data(self):
        """Load PDF files from data directory."""
        self.raw_data = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        self.logger.info(f"Found {len(self.raw_data)} PDF files")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Extract text from PDFs using pdfplumber and OCR."""
        self.processed_data = []

        # TODO: Implement PDF text extraction
        # for pdf_file in self.raw_data:
        #     text = ""
        #     try:
        #         # Try text-based extraction first
        #         with pdfplumber.open(pdf_file) as pdf:
        #             for page in pdf.pages:
        #                 page_text = page.extract_text()
        #                 if page_text:
        #                     text += page_text + "\n"
        #
        #         # If no text found, try OCR (scanned PDF)
        #         if not text.strip():
        #             images = convert_from_path(pdf_file)
        #             for img in images:
        #                 text += pytesseract.image_to_string(img) + "\n"
        #
        #         self.processed_data.append({
        #             "file": pdf_file,
        #             "text": text,
        #         })
        #     except Exception as e:
        #         self.logger.error(f"Error processing {pdf_file}: {e}")

        self.logger.info(f"Processed {len(self.processed_data)} PDF files")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from PDF text using NER."""
        self.extracted_records = []

        # TODO: Implement extraction logic
        # nlp = spacy.load("en_core_web_sm")
        #
        # for idx, item in enumerate(self.processed_data, start=1):
        #     doc = nlp(item["text"])
        #
        #     # Extract entities
        #     persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        #     locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
        #     dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        #     orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        #
        #     record = {
        #         "Incident_ID": generate_incident_id(idx),
        #         "Source": self.SOURCE_LABEL,
        #         "Report_ID": f"RPT_{idx:03d}",
        #         "Department": orgs[0] if orgs else "",
        #         "Doc_Type": "TODO: classify document type",
        #         "Date": dates[0] if dates else "",
        #         "Location": ", ".join(locations) if locations else "",
        #         "Officer": persons[0] if persons else "",
        #         "Summary": item["text"][:300],  # First 300 chars
        #         "Severity": classify_severity(item["text"]),
        #     }
        #     self.extracted_records.append(record)

        self.logger.info(f"Extracted {len(self.extracted_records)} records")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = PDFPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print(results)
