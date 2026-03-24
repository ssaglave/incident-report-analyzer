"""
Student 2 — Document Analyst
Processes police report PDFs using pdfplumber for text extraction,
pytesseract for OCR on scanned docs, and spaCy for NER.

Uses the COMMON pipeline: pipeline/base_pipeline.py
"""

import os
import sys
import glob
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import BasePipeline, generate_incident_id, classify_severity

# ── Import PDF & NER libraries ──
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import spacy


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
        self.raw_data.sort()  # Sort for consistent ordering
        self.logger.info(f"Found {len(self.raw_data)} PDF files")

    # ── STAGE 2: AI Processing ──────────────────────────────────────────────

    def process_data(self):
        """Extract text from PDFs using pdfplumber and OCR."""
        self.processed_data = []

        for pdf_file in self.raw_data:
            text = ""
            try:
                # Try text-based extraction first using pdfplumber
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

                # If no text found, try OCR (scanned PDF)
                if not text.strip():
                    self.logger.info(f"No text found in {os.path.basename(pdf_file)}, attempting OCR...")
                    try:
                        images = convert_from_path(pdf_file)
                        for img in images:
                            text += pytesseract.image_to_string(img) + "\n"
                    except Exception as ocr_error:
                        self.logger.warning(f"OCR failed for {pdf_file}: {ocr_error}")

                if text.strip():
                    self.processed_data.append({
                        "file": pdf_file,
                        "text": text,
                    })
                    self.logger.info(f"✓ Extracted text from {os.path.basename(pdf_file)}")
                else:
                    self.logger.warning(f"No text extracted from {pdf_file}")

            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {e}")

        self.logger.info(f"Processed {len(self.processed_data)} PDF files")

    # ── STAGE 3: Information Extraction ─────────────────────────────────────

    def extract_information(self):
        """Extract structured fields from PDF text using spaCy NER."""
        self.extracted_records = []

        try:
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return

        for idx, item in enumerate(self.processed_data, start=1):
            try:
                doc = nlp(item["text"][:100000])  # Limit text length for spaCy

                # Extract entities using NER
                persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
                dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

                # Enhanced location extraction: also try to find addresses with streets
                full_location = self._extract_location(item["text"], locations)

                # Classify document type based on keywords
                doc_type = self._classify_document_type(item["text"])

                # Extract case/report number using regex
                report_id = self._extract_report_id(item["text"])

                # Create record
                record = {
                    "Incident_ID": generate_incident_id(idx),
                    "Source": self.SOURCE_LABEL,
                    "Report_ID": report_id,
                    "Department": orgs[0] if orgs else "",
                    "Doc_Type": doc_type,
                    "Date": dates[0] if dates else "",
                    "Location": full_location,
                    "Officer": persons[0] if persons else "",
                    "Summary": item["text"][:300],  # First 300 chars
                    "Severity": classify_severity(item["text"]),
                }
                self.extracted_records.append(record)
                self.logger.info(f"✓ Extracted record {idx}: {doc_type} - {record['Severity']} severity - Location: {full_location}")

            except Exception as e:
                self.logger.error(f"Error extracting from record {idx}: {e}")

        self.logger.info(f"Extracted {len(self.extracted_records)} records")

    # ── Helper Methods ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_location(text: str, ner_locations: list) -> str:
        """
        Extract location with enhanced logic.
        First tries to find full street addresses, then falls back to NER results.
        """
        # Pattern 1: Full address with street number and city/state
        # e.g., "1847 Oak Avenue, Springdale, AR 72764"
        address_pattern = r"(\d+\s+[\w\s]+(?:Street|Street|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Circle|Dr|Drive|Place|Pl|Way),?\s+[\w\s]+,?\s+\w{2}\s+\d{5})"
        match = re.search(address_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: Street address without zip
        # e.g., "1847 Oak Avenue, Springdale, AR"
        address_pattern2 = r"(\d+\s+[\w\s]+(?:Street|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd),?\s+[\w\s]+,?\s+\w{2})"
        match = re.search(address_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: Interstate/Highway locations
        # e.g., "Interstate 49 at Exit 62"
        highway_pattern = r"((?:Interstate|US|Highway|Route|I-|US-)\s*\d+(?:\s+at\s+Exit\s+\d+)?)"
        match = re.search(highway_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback to NER results
        if ner_locations:
            return ", ".join(ner_locations[:2])

        return ""

    @staticmethod
    def _classify_document_type(text: str) -> str:
        """Classify document type based on keywords in text."""
        text_lower = text.lower()

        doc_types = {
            "Burglary": ["burglary", "break-in", "breaking and entering"],
            "Theft": ["theft", "larceny", "stolen", "shoplifting"],
            "Accident": ["motor vehicle accident", "accident", "collision", "crash"],
            "Assault": ["assault", "battery", "fight"],
            "Report": ["report", "incident report"],
            "Training": ["training", "proposal", "1033 program"],
        }

        for doc_type, keywords in doc_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type

        return "Report"  # Default classification

    @staticmethod
    def _extract_report_id(text: str) -> str:
        """Extract case/report number from text using regex patterns."""
        # Common patterns: Case Number: 2024-001234 or Report ID: RPT_001
        patterns = [
            r"Case\s*Number[:\s]+(\d{4}-\d+)",
            r"Report\s*ID[:\s]+(\w+_?\d+)",
            r"Case\s*#[:\s]+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return "RPT_UNKNOWN"  # Default if no match


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = PDFPipeline(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )
    results = pipeline.run()
    print("\n" + "="*60)
    print("PDF ANALYSIS RESULTS:")
    print("="*60)
    print(results)
