# Multimodal Crime / Incident Report Analyzer — Project Report

---

## 1. Introduction

This project implements an **AI-powered multimodal incident report analyzer** that processes five types of unstructured data — audio, PDF documents, images, video, and text — and integrates them into a single, unified incident report dataset. The system is designed to assist emergency response teams by automatically extracting, classifying, and correlating incident information across different data modalities.

Each modality is processed through a standardized 5-stage pipeline:

1. **Stage 1 — Unstructured Data Ingestion**: Load raw data files
2. **Stage 2 — AI Processing per Modality**: Run AI/ML models for analysis
3. **Stage 3 — Information Extraction**: Extract structured fields from AI output
4. **Stage 4 — Structured Dataset Generation**: Output to standardized CSV
5. **Stage 5 — Dashboard / Query System**: Visualization and querying (Streamlit)

A shared `BasePipeline` abstract class (`pipeline/base_pipeline.py`) enforces consistency across all modules, ensuring every output includes `Incident_ID`, `Source`, and `Severity` columns.

---

## 2. Audio Module (Student 1)

### 2.1 Approach

The audio module processes emergency 911 call recordings to extract incident events, locations, sentiment, and urgency levels. It supports three data ingestion modes:

- **Mode A (Kaggle)**: Loads `911_metadata.csv` with linked `.wav` files; uses OpenAI Whisper for speech-to-text transcription combined with metadata descriptions
- **Mode B (CSV)**: Pre-transcribed text loaded directly, skipping Whisper
- **Mode C (Audio)**: Raw `.wav`/`.mp3` files transcribed entirely with Whisper

The pipeline performs keyword-based event classification across 10 incident categories (fire, shooting, robbery, medical emergency, etc.), calculates a weighted urgency score (60% keyword-based, 40% sentiment-based), and uses spaCy NER to extract locations and persons.

### 2.2 Tools & Libraries

| Tool | Purpose |
|------|---------|
| **OpenAI Whisper** (`whisper`, base model) | Speech-to-text transcription of audio recordings |
| **spaCy** (`en_core_web_sm`) | Named Entity Recognition (GPE, LOC, FAC, PERSON) |
| **HuggingFace Transformers** (`distilbert-base-uncased-finetuned-sst-2-english`) | Sentiment analysis (Distressed/Calm classification) |
| **pandas** | Data manipulation and CSV output |

### 2.3 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | 911 Recordings: The First 6 Seconds |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds) |
| **Format** | CSV metadata (`911_metadata.csv`) + `.wav` audio files |
| **Size** | 712 records with metadata (title, date, state, deaths, description) |
| **Fields** | filename, title, date, state, deaths, potential_death, false_alarm, description, event_id |

### 2.4 Challenges

- **Short audio clips** (6 seconds): Whisper transcriptions are often incomplete or noisy for very short recordings, requiring fallback to metadata descriptions
- **Combining multiple text sources**: Needed to merge Whisper transcriptions with Kaggle metadata descriptions to get sufficient context for event classification
- **Urgency scoring calibration**: Balancing keyword-based urgency with sentiment analysis scores required iterative tuning of the 60/40 weighting
- **Ambiguous events**: Some transcripts matched multiple event categories; resolved by using a scoring system that selects the highest-scoring match

### 2.5 Results

| Metric | Value |
|--------|-------|
| **Records Produced** | 701 |
| **Output File** | `audio/output/audio_results.csv` |
| **Severity — High** | 421 (60.1%) |
| **Severity — Medium** | 218 (31.1%) |
| **Severity — Low** | 62 (8.8%) |
| **Output Columns** | Incident_ID, Source, Call_ID, Transcript, Extracted_Event, Location, Timestamp, Sentiment, Urgency_Score, Severity |

**Event Types Detected**: Building fire / trapped persons, Murder / homicide, Shooting, Road accident, Medical emergency, Assault / violence, Domestic disturbance, Robbery / theft, Public disturbance, Kidnapping / missing, Unknown incident

---

## 3. PDF Module (Student 2)

### 3.1 Approach

The PDF module processes police incident report documents. It uses a two-tier text extraction strategy:

1. **Primary**: `pdfplumber` for text-based PDFs (extracts text directly from the PDF layer)
2. **Fallback OCR**: `pdf2image` + `pytesseract` for scanned/image-based PDFs

After extraction, spaCy NER identifies persons (officers), organizations (departments), dates, and locations. Regex patterns extract structured fields like case numbers and full street addresses. Document type classification uses keyword matching against categories like Burglary, Theft, Accident, Assault, and Training.

### 3.2 Tools & Libraries

| Tool | Purpose |
|------|---------|
| **pdfplumber** | Text extraction from digital PDFs |
| **pytesseract** + **pdf2image** | OCR for scanned PDF documents |
| **spaCy** (`en_core_web_sm`) | NER for persons, organizations, dates, locations |
| **Regex patterns** | Case number extraction, street address parsing |

### 3.3 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Arkansas PD 1033 Training Proposals & Police Reports |
| **Source** | [MuckRock](https://www.muckrock.com) — FOIA-obtained police documents |
| **Format** | `.pdf` files |
| **Size** | 3 PDF documents |
| **Content** | Burglary reports, motor vehicle accident reports, training proposals |

### 3.4 Challenges

- **Mixed PDF formats**: Some PDFs contained digital text while others were scanned images, requiring the two-tier extraction approach
- **Variable document structure**: Police reports from different departments follow different formats; the regex-based extraction needed multiple patterns for addresses and case numbers
- **Small dataset**: Only 3 PDF documents were available, limiting the statistical significance of results
- **OCR accuracy**: Scanned documents sometimes produced noisy text that affected NER accuracy

### 3.5 Results

| Metric | Value |
|--------|-------|
| **Records Produced** | 3 |
| **Output File** | `pdf/output/pdf_results.csv` |
| **Severity — Medium** | 3 (100%) |
| **Output Columns** | Incident_ID, Source, Report_ID, Department, Doc_Type, Date, Location, Officer, Summary, Severity |

**Document Types Identified**: Burglary, Accident, Assault

---

## 4. Image Module (Student 3)

### 4.1 Approach

The image module analyzes incident scene photographs using a combination of object detection, scene classification, and OCR:

1. **Object Detection**: Reads pre-annotated YOLO-format label files bundled with the Roboflow dataset. Falls back to live YOLOv8 inference if no labels are found.
2. **Scene Classification**: Uses a pretrained ResNet18 (ImageNet) classifier to predict scene types, with a rule-based fallback mapping detected objects to scene categories (Fire Scene, Accident Scene, Crime Scene, Emergency Response Scene).
3. **OCR**: Optional `pytesseract` text extraction for images containing text (signs, placards).
4. **Annotated Output**: Draws bounding boxes on detected objects and saves annotated copies.

### 4.2 Tools & Libraries

| Tool | Purpose |
|------|---------|
| **YOLO Label Files** (pre-annotated) | Object detection from bundled dataset annotations |
| **YOLOv8** (`ultralytics`, optional) | Live object detection when labels unavailable |
| **ResNet18** (`torchvision`) | Pretrained scene classification |
| **pytesseract** (optional) | OCR text extraction from images |
| **Pillow (PIL)** | Image loading, annotated image generation |

### 4.3 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Fire Dataset for YOLOv8 — v2 |
| **Source** | [Roboflow Universe](https://universe.roboflow.com/fire-detection-dataset/fire-dataset-for-yolov8-vbljx/dataset/2) |
| **License** | CC BY 4.0 |
| **Format** | YOLO format (images + label `.txt` files + `data.yaml`) |
| **Size** | 3,996 images across train/valid/test splits |
| **Classes** | 2 classes — `0` (non-fire), `1` (fire) |
| **Annotations** | 3,996 corresponding label files with bounding box coordinates |
| **No API Key Required** | ✅ Dataset is downloaded directly, no Roboflow API needed |

### 4.4 Challenges

- **Label path resolution**: The project's module lives in an `images/` directory, and the YOLO dataset also uses `images/` subdirectories. The label path algorithm initially replaced the wrong `/images/` segment. Fixed by using `rsplit` to replace only the last occurrence.
- **Missing `Incident_ID` column**: The original code used `Image_ID` without `Incident_ID`, `Source`, or `Severity`, which are required by the integration pipeline. Added these fields using the shared `generate_incident_id()` and `classify_severity()` utilities.
- **Class name mapping**: The `data.yaml` maps class `0` to the literal string `'0'`, which was originally treated as `"unknown"`. Fixed to map to `"non-fire"` for clarity.
- **Large dataset processing**: 3,996 images needed efficient label-based detection (reading `.txt` files) rather than running live YOLO inference on each image.

### 4.5 Results

| Metric | Value |
|--------|-------|
| **Records Produced** | 3,996 |
| **Output File** | `images/output/image_results.csv` |
| **Severity — High** | 3,940 (98.6%) — fire detected |
| **Severity — Low** | 56 (1.4%) — no fire objects |
| **Annotated Images** | 100 saved in `images/output/annotated/` |
| **Output Columns** | Incident_ID, Source, Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Confidence, Severity |

**Scene Types Identified**: Fire Scene (3,411), General Incident Scene (529), Unknown (56)

**Objects_Detected Format**: `fire (1.00)` — includes object names with average confidence score in parentheses.

---

## 5. Video Module (Student 4)

### 5.1 Approach

The video module analyzes CCTV surveillance footage by:

1. **Frame Sampling**: Extracts every 10th frame from video files to balance coverage with processing speed
2. **Object Detection**: Uses YOLOv8 for detecting objects (persons, vehicles, fire, weapons); falls back to a mock detector with randomized results if `ultralytics` is unavailable
3. **Motion Analysis**: OpenCV MOG2 background subtraction to compute motion scores
4. **Incident Classification**: Rule-based system matching detected object combinations to incident types (e.g., `{fire, smoke}` → "Fire", `{car, truck}` → "Vehicle Crash") with severity levels
5. **Frame Annotation**: Saves annotated key frames with bounding boxes and event labels

### 5.2 Tools & Libraries

| Tool | Purpose |
|------|---------|
| **OpenCV** (`cv2`) | Video reading, frame extraction, MOG2 motion analysis |
| **YOLOv8** (`ultralytics`, optional) | Real-time object detection on video frames |
| **NumPy** | Array operations for frame processing |
| **Mock detector** (fallback) | Simulated detections when YOLO unavailable |

### 5.3 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | CAVIAR CCTV Dataset |
| **Source** | [University of Edinburgh](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) |
| **Format** | `.avi` video files |
| **Size** | 4 synthetic CCTV-style clips (320×240, 25fps, 5s each) |
| **Content** | Simulated corridor and entrance surveillance footage |
| **Generation** | Auto-generated using OpenCV if real CAVIAR clips not present |

### 5.4 Challenges

- **Output filename mismatch**: The video analyzer script saved to `video_analysis.csv` but the integration pipeline expected `video_results.csv`. Resolved by renaming the output file.
- **Incident_ID format**: The notebook (`video_analyst.ipynb`) generated hex UUID-style IDs (e.g., `B7F145D5`) instead of the `INC_XXX` sequential format used by other modules. Fixed by reassigning IDs to match the standard format.
- **Dual implementations**: Both a `.py` script and a Jupyter notebook existed with different column schemas. The notebook-generated CSV had the correct integration-compatible columns.
- **Synthetic data limitations**: Without real CAVIAR dataset clips, the system generates synthetic videos with simulated motion patterns, which limits detection diversity.

### 5.5 Results

| Metric | Value |
|--------|-------|
| **Records Produced** | 52 |
| **Output File** | `video/output/video_results.csv` |
| **Severity — Medium** | 42 (80.8%) |
| **Severity — Critical** | 4 (7.7%) |
| **Severity — High** | 4 (7.7%) |
| **Severity — Low** | 2 (3.8%) |
| **Annotated Frames** | Saved in `video/output/frames/` |
| **Output Columns** | Incident_ID, Source_File, Frame_Number, Timestamp_Sec, Detected_Objects, Object_Count, Motion_Score, Video_Event, Severity, Confidence, Frame_Saved, Analyzed_At |

**Video Events Detected**: Suspicious Motion (42), Fire (4), Vehicle Crash (4), No Incident (2)

---

## 6. Text Module (Student 5)

### 6.1 Approach

The text module processes crime-related text reports (Twitter/social media data and CSV records) using an NLP pipeline:

1. **Text Cleaning**: Removes URLs, normalizes whitespace, filters retweets and short texts (<4 words)
2. **Named Entity Recognition**: spaCy identifies locations (GPE, LOC, FAC), persons, and organizations
3. **Sentiment Analysis**: HuggingFace `distilbert` classifies text as POSITIVE or NEGATIVE
4. **Zero-Shot Topic Classification**: Facebook's `bart-large-mnli` model classifies text into 8 crime categories without task-specific training data
5. **Crime Type Resolution**: Uses provided labels if available, otherwise falls back to the zero-shot classifier's top prediction

### 6.2 Tools & Libraries

| Tool | Purpose |
|------|---------|
| **spaCy** (`en_core_web_sm`) | Named Entity Recognition |
| **HuggingFace Transformers** (`distilbert-base-uncased-finetuned-sst-2-english`) | Sentiment analysis |
| **HuggingFace Transformers** (`facebook/bart-large-mnli`) | Zero-shot topic/crime type classification |
| **pandas** | CSV data loading and manipulation |

### 6.3 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | CrimeReport Dataset (Twitter crime reports) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport) |
| **Format** | JSON-lines `.txt` file (one JSON object per line) |
| **Size** | 115 records (80 after filtering retweets and short texts) |
| **Content** | Crime-related tweets with geo-location, timestamps, user metadata |
| **Fields** | text, created_at, place (full_name), id, lang, retweet_count |

### 6.4 Challenges

- **Short and noisy text**: Social media posts are informal, contain abbreviations, URLs, and hashtags that need cleaning before NLP processing
- **Retweet filtering**: Duplicate content from retweets needed to be filtered to avoid inflated record counts
- **Zero-shot classification latency**: The `bart-large-mnli` model is computationally expensive for each record; processing 80 texts requires significant time
- **Location extraction**: Many tweets lack geolocation data; the pipeline falls back to NER-extracted location entities from the text itself
- **Topic label quality**: Zero-shot classification occasionally misclassifies crime types; the 8                 predefined categories (shooting, robbery, theft, fire, assault, traffic incident, police investigation, public disturbance) cover most but not all incident types

### 6.5 Results

| Metric | Value |
|--------|-------|
| **Records Produced** | 80 |
| **Output File** | `text/output/text_results.csv` |
| **Severity — High** | 49 (61.3%) |
| **Severity — Medium** | 26 (32.5%) |
| **Severity — Low** | 5 (6.3%) |
| **Output Columns** | Incident_ID, Text_ID, Crime_Type, Location_Entity, Sentiment, Topic, Severity |

**Crime Types Identified**: shooting, police investigation, robbery, public disturbance, fire, assault, theft, traffic incident

---

## 7. Integration Pipeline

### 7.1 Approach

The integration pipeline (`integration/merge_pipeline.py`) merges all 5 module outputs into a single curated dataset following a 5-step process:

1. **Step 1 — Load Module Outputs**: Reads all 5 CSV files from each module's output directory
2. **Step 2 — Normalize Incident IDs & Merge**: Standardizes `Incident_ID` to `INC_XXX` format; performs outer joins on `Incident_ID` to combine all modalities
3. **Step 3 — Handle Missing Values**: Fills NaN/empty values with `"N/A"` where a modality has no data for a given incident
4. **Step 4 — Compute Overall Severity**: Parses each module's severity signal (text labels or numeric 0–1 scores), takes the maximum across all modalities, and applies an escalation rule (if 2+ modules report High/Critical, bump severity up by one level, capped at Critical)
5. **Step 5 — Finalize Output**: Arranges columns in the required unified schema order and saves to CSV

### 7.2 Column Mapping

| Unified Column | Source Module | Source Column |
|----------------|--------------|---------------|
| `Incident_ID` | All modules | `Incident_ID` (common key) |
| `Audio_Event` | Audio | `Extracted_Event` |
| `PDF_Doc_Type` | PDF | `Doc_Type` |
| `Image_Objects` | Image | `Objects_Detected` |
| `Video_Event` | Video | `Video_Event` |
| `Text_Crime_Type` | Text | `Crime_Type` |
| `Severity` | Combined | Max across all modules + escalation |

### 7.3 Severity Computation Logic

```
For each incident:
  1. Parse each module's severity → numeric rank (Low=1, Medium=2, High=3, Critical=4)
  2. For numeric scores (0–1): ≥0.75→High, ≥0.45→Medium, <0.45→Low
  3. Take the maximum rank across all available modalities
  4. If 2+ modules report High or Critical → escalate by +1 rank (cap at Critical)
  5. Map final rank back to label
```

### 7.4 Challenges

- **Heterogeneous ID formats**: Video module used hex UUIDs while others used `INC_XXX`. Solved by normalizing all IDs to sequential format before merging.
- **Unequal dataset sizes**: Audio (701), Image (3,996), Text (80), Video (52), PDF (3) — the outer join produces rows where most modalities show `N/A`. The image module dominates the final row count.
- **Severity signal diversity**: Audio uses numeric urgency scores (0–1), image uses confidence scores, while PDF/video/text use categorical labels. The `_parse_severity()` function handles both formats.
- **Video output filename**: The video module wrote to `video_analysis.csv` but the integration expected `video_results.csv`. Resolved by renaming.
- **Image module missing required columns**: The original image output lacked `Incident_ID` and `Severity`. Fixed by updating the image analyzer to include these fields.

### 7.5 Results

| Metric | Value |
|--------|-------|
| **Total Records** | 3,996 |
| **Output File** | `integration/output/final_incident_report.csv` |
| **File Size** | 149 KB |
| **Columns** | 7 (Incident_ID, Audio_Event, PDF_Doc_Type, Image_Objects, Video_Event, Text_Crime_Type, Severity) |

**Severity Distribution:**

| Severity | Count | Percentage |
|----------|-------|------------|
| High | 3,589 | 89.8% |
| Critical | 356 | 8.9% |
| Low | 49 | 1.2% |
| Medium | 2 | 0.05% |

**Data Coverage per Modality:**

| Modality | Rows with Data | Coverage |
|----------|---------------|----------|
| Audio_Event | 701 | 17.5% |
| PDF_Doc_Type | 3 | 0.1% |
| Image_Objects | 3,940 | 98.6% |
| Video_Event | 52 | 1.3% |
| Text_Crime_Type | 80 | 2.0% |

**Multi-modal Rows (data from 3+ modalities):** 79 incidents

### 7.6 Sample Output

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001 | Building fire / trapped persons | Burglary | fire (1.00) | Suspicious Motion | shooting | Critical |
| INC_002 | Murder / homicide | Accident | fire (1.00) | Suspicious Motion | police investigation | Critical |
| INC_003 | Shooting | Assault | fire (1.00) | Fire | police investigation | Critical |

---

## 8. Pipeline Architecture

The system follows a modular architecture where each student's module is an independent subclass of `BasePipeline`:

```
┌────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                          │
│                (Master Pipeline Runner)                     │
├────────┬────────┬────────┬────────┬────────┐               │
│ Audio  │  PDF   │ Image  │ Video  │  Text  │               │
│Pipeline│Pipeline│Pipeline│Pipeline│Pipeline│               │
│  (S1)  │  (S2)  │  (S3)  │  (S4)  │  (S5)  │               │
├────────┴────────┴────────┴────────┴────────┤               │
│            BasePipeline (Abstract)          │               │
│    load_data → process_data → extract →    │               │
│              generate_output               │               │
└────────────────┬───────────────────────────┘               │
                 │  5 CSV Outputs                             │
                 ▼                                            │
┌────────────────────────────────────────────┐               │
│        merge_pipeline.py                   │               │
│   Load → Normalize → Merge → Severity →   │               │
│           Finalize & Save                  │               │
└────────────────┬───────────────────────────┘               │
                 │                                            │
                 ▼                                            │
    integration/output/final_incident_report.csv              │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Technology Stack Summary

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Speech-to-Text** | OpenAI Whisper (base) |
| **NLP / NER** | spaCy (en_core_web_sm) |
| **Sentiment Analysis** | HuggingFace distilbert-base-uncased-finetuned-sst-2-english |
| **Topic Classification** | HuggingFace facebook/bart-large-mnli (zero-shot) |
| **Object Detection** | YOLOv8 (ultralytics) + YOLO label files |
| **Scene Classification** | ResNet18 (torchvision, pretrained ImageNet) |
| **PDF Processing** | pdfplumber, pytesseract, pdf2image |
| **Video Processing** | OpenCV (cv2), MOG2 background subtraction |
| **Image Processing** | Pillow (PIL) |
| **Data Processing** | pandas, NumPy |
| **Pipeline Framework** | Custom BasePipeline (abstract base class) |

---

## 10. Conclusion

The Multimodal Incident Report Analyzer successfully demonstrates end-to-end processing of five distinct data modalities into a unified incident report. Key achievements:

- **3,996 incident records** in the final curated dataset with a standardized 7-column schema
- **79 multi-modal records** where 3+ data sources contribute to the same incident
- **Automated severity classification** combining signals from all available modalities with escalation logic
- **No API keys required** — all processing uses locally-installed models and open datasets
- **Modular architecture** allowing independent development and testing of each modality through the shared `BasePipeline` framework

The project demonstrates practical application of AI/ML techniques (speech recognition, NER, sentiment analysis, object detection, zero-shot classification) to real-world emergency response data processing problems.

---

*Report generated: March 31, 2026*
*Repository: https://github.com/ssaglave/incident-report-analyzer*
