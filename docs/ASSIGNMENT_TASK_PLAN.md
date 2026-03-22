# 🚨 Multimodal Crime / Incident Report Analyzer — Task Plan

## Project Timeline: 2 Weeks

---

## 📅 Phase Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1** | Days 1–2 | Setup, Dataset Collection, Environment Config |
| **Phase 2** | Days 3–8 | Individual Module Development (per student) |
| **Phase 3** | Days 9–11 | Integration & Unified Pipeline |
| **Phase 4** | Days 12–13 | Dashboard, Documentation, Report |
| **Phase 5** | Day 14 | Final Demo Preparation & Submission |

---

## 🔧 Phase 1 — Project Setup (Days 1–2) — **All Students**

> [!IMPORTANT]
> Every student must complete these setup tasks before starting individual work.

- [ ] Create a shared **GitHub repository** with folder structure:
  ```
  /audio      → Student 1
  /pdf        → Student 2
  /images     → Student 3
  /video      → Student 4
  /text       → Student 5
  /integration → Team Lead / All
  /dashboard   → Team Lead / All
  /docs        → Reports, diagrams
  README.md
  requirements.txt
  ```
- [ ] Set up a **shared Python virtual environment** and agree on Python version (3.10+ recommended)
- [ ] Create `requirements.txt` with common dependencies:
  ```
  pandas
  spacy
  transformers
  openai-whisper
  pymupdf
  pdfplumber
  pytesseract
  ultralytics
  opencv-python
  moviepy
  nltk
  streamlit
  ```
- [ ] **Agree on a common output schema** — every student's CSV must include:
  - `Incident_ID` (shared key for merging)
  - `Source` (audio / pdf / image / video / text)
  - `Event` / `Event_Detected`
  - `Location`
  - `Timestamp` or `Date`
  - `Severity` (Low / Medium / High)
- [ ] Each student **downloads their dataset** (links in assignment brief)
- [ ] Create the **AI Pipeline Architecture Diagram** (use draw.io, Lucidchart, or Mermaid)

---

## 👤 Phase 2 — Individual Module Development (Days 3–8)

Each student works independently on their module. Every module follows the same 3-stage pattern:

```
Stage A: Data Ingestion & Preprocessing
Stage B: AI Model Processing & Extraction
Stage C: Structured Output Generation (CSV)
```

---

### 🎙️ Student 1 — Audio Analyst

**Goal:** Convert emergency audio calls → structured CSV

#### Stage A: Data Ingestion (Day 3)
- [ ] Download 911 Calls dataset from Kaggle ([link](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2))
- [ ] Fork the Wav2Vec2 notebook on Kaggle (click "Copy and Edit")
- [ ] Run the pre-built transcription notebook to verify working audio → text pipeline
- [ ] Load audio files into a processing script; handle file formats (.wav, .mp3)

#### Stage B: AI Processing (Days 4–6)
- [ ] **Speech-to-Text:** Use `openai-whisper` to transcribe audio files
  ```python
  import whisper
  model = whisper.load_model("base")
  result = model.transcribe("audio_file.wav")
  transcript = result["text"]
  ```
- [ ] **Keyword Extraction:** Use `spaCy` NER to extract:
  - Incident type (fire, accident, robbery, etc.)
  - Location mentions (street names, landmarks)
  - Person names
  - Urgency phrases ("help!", "trapped", "hurry")
- [ ] **Sentiment / Urgency Analysis:** Use HuggingFace `transformers`
  ```python
  from transformers import pipeline
  sentiment = pipeline("sentiment-analysis")
  result = sentiment(transcript)
  ```
- [ ] Calculate an **Urgency Score** (0.0–1.0) based on sentiment + keyword presence

#### Stage C: Structured Output (Days 7–8)
- [ ] Generate output CSV with columns:
  | Call_ID | Transcript | Extracted_Event | Location | Sentiment | Urgency_Score |
  |---------|-----------|-----------------|----------|-----------|---------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Validate output: check for nulls, normalize text fields
- [ ] Save to `/audio/output/audio_results.csv`

---

### 📄 Student 2 — Document Analyst

**Goal:** Extract structured data from police report PDFs → structured CSV

#### Stage A: Data Ingestion (Day 3)
- [ ] Download Arkansas Police Department 1033 Training PDF from [MuckRock](https://www.muckrock.com)
- [ ] Collect 2–3 additional sample police report PDFs (search for public FOIA docs)
- [ ] Categorize PDFs: text-based vs. scanned (image-based)

#### Stage B: AI Processing (Days 4–6)
- [ ] **Text Extraction (text-based PDFs):** Use `pdfplumber`
  ```python
  import pdfplumber
  with pdfplumber.open("report.pdf") as pdf:
      for page in pdf.pages:
          text = page.extract_text()
  ```
- [ ] **OCR (scanned PDFs):** Use `pytesseract`
  ```python
  from pdf2image import convert_from_path
  import pytesseract
  images = convert_from_path("scanned_report.pdf")
  for img in images:
      text = pytesseract.image_to_string(img)
  ```
- [ ] **Named Entity Recognition:** Use `spaCy` to extract:
  - Department name
  - Document type / category
  - Date
  - Program / key details
  - Officer names, suspect descriptions
- [ ] **Table Extraction:** Use `pdfplumber` to extract embedded tables

#### Stage C: Structured Output (Days 7–8)
- [ ] Generate output CSV with columns:
  | Report_ID | Department | Doc_Type | Date | Program | Key_Detail |
  |-----------|-----------|----------|------|---------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Handle edge cases: multi-page reports, inconsistent formats
- [ ] Save to `/pdf/output/pdf_results.csv`

---

### 🖼️ Student 3 — Image Analyst

**Goal:** Detect objects and classify scenes in incident photos → structured CSV

#### Stage A: Data Ingestion (Day 3)
- [ ] Create a Roboflow account (free tier)
- [ ] Download fire detection dataset from [Roboflow](https://universe.roboflow.com) in YOLOv8 format
- [ ] Choose a dataset with **1000+ images** and a trained model badge
- [ ] Organize images into `/images/data/` directory

#### Stage B: AI Processing (Days 4–6)
- [ ] **Object Detection:** Use `YOLOv8` (ultralytics)
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")  # or fine-tuned model
  results = model("scene_image.jpg")
  ```
- [ ] Extract from results:
  - Detected objects (fire, smoke, vehicle, person, weapon)
  - Bounding box coordinates
  - Confidence scores
- [ ] **Scene Classification:** Use a pre-trained classifier (ResNet / ViT from HuggingFace)
  - Classify scene type: accident, fire, theft, public disturbance
- [ ] **OCR on images:** Use `pytesseract` for text in images
  - Extract license plates, street signs, building names

#### Stage C: Structured Output (Days 7–8)
- [ ] Generate output CSV with columns:
  | Image_ID | Scene_Type | Objects_Detected | Bounding_Boxes | Confidence |
  |----------|-----------|------------------|----------------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save sample annotated images to `/images/output/annotated/`
- [ ] Save to `/images/output/image_results.csv`

---

### 🎥 Student 4 — Video Analyst

**Goal:** Process CCTV footage to detect events and produce timestamped logs → structured CSV

#### Stage A: Data Ingestion (Day 3)
- [ ] Download 3–5 short clips from [CAVIAR Dataset](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- [ ] Recommended folders: `Browse`, `OneStopEnter`, `Fight`, `Collapse`
- [ ] Convert `.mpg` files to `.mp4` if needed (use `ffmpeg` or `moviepy`)

#### Stage B: AI Processing (Days 4–6)
- [ ] **Frame Extraction:** Use `OpenCV` to extract frames at intervals
  ```python
  import cv2
  cap = cv2.VideoCapture("clip.mp4")
  fps = cap.get(cv2.CAP_PROP_FPS)
  # Extract 1 frame per second
  frame_interval = int(fps)
  ```
- [ ] **Motion Detection:** Compare consecutive frames using:
  - Frame differencing
  - Background subtraction (MOG2 algorithm)
  - Flag frames with significant motion changes
- [ ] **Object/Activity Detection:** Run `YOLOv8` on extracted key frames
  - Detect: persons, fighting, collapsing, vehicles
  - Count persons per frame
- [ ] **Anomaly Flagging:** Mark events that deviate from baseline activity
  - Sudden motion spikes → potential incident
  - Person count changes → crowd gathering or fleeing

#### Stage C: Structured Output (Days 7–8)
- [ ] Generate output CSV with columns:
  | Clip_ID | Timestamp | Frame_ID | Event_Detected | Persons_Count | Confidence |
  |---------|-----------|----------|----------------|---------------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save key frames as images to `/video/output/frames/`
- [ ] Save to `/video/output/video_results.csv`

---

### 📝 Student 5 — Text Analyst

**Goal:** Analyze crime text reports using NLP → structured CSV

#### Stage A: Data Ingestion (Day 3)
- [ ] Download CrimeReport dataset from [Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
- [ ] Load CSV: `df = pd.read_csv('crimereport.csv')`
- [ ] Explore dataset: check columns, data types, missing values, sample rows

#### Stage B: AI Processing (Days 4–6)
- [ ] **Text Preprocessing:** Use `NLTK` / `spaCy`
  - Remove noise (special characters, URLs, extra whitespace)
  - Tokenize text
  - Remove stopwords
  - Normalize (lowercasing, lemmatization)
- [ ] **Named Entity Recognition:** Use `spaCy`
  ```python
  import spacy
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(crime_text)
  entities = [(ent.text, ent.label_) for ent in doc.ents]
  ```
  - Extract: person names, locations, organizations, dates
- [ ] **Sentiment Analysis:** Use HuggingFace `transformers`
  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis")
  result = classifier(crime_text)
  ```
- [ ] **Topic Classification:** Use zero-shot classification
  ```python
  from transformers import pipeline
  classifier = pipeline("zero-shot-classification")
  labels = ["accident", "fire", "theft", "robbery", "assault", "public disturbance"]
  result = classifier(crime_text, candidate_labels=labels)
  ```
- [ ] **Severity Labeling:** Rule-based or model-based severity assignment
  - Keywords like "murder", "fire", "weapon" → High
  - "theft", "vandalism" → Medium
  - "noise complaint", "minor" → Low

#### Stage C: Structured Output (Days 7–8)
- [ ] Generate output CSV with columns:
  | Text_ID | Crime_Type | Location_Entity | Sentiment | Topic | Severity_Label |
  |---------|-----------|-----------------|-----------|-------|----------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save to `/text/output/text_results.csv`

---

## 🔗 Phase 3 — Integration & Unified Pipeline (Days 9–11) — **All Students**

> [!IMPORTANT]
> This is the **most critical deliverable**. All 5 outputs merge here.

### Step 1: Define Common Incident_ID (Day 9)
- [ ] Agree on Incident_ID format: `INC_001`, `INC_002`, etc.
- [ ] Each student maps their records to shared Incident_IDs
- [ ] Create a mapping file: `incident_mapping.csv`

### Step 2: Merge DataFrames (Day 9–10)
- [ ] Write integration script: `/integration/merge_pipeline.py`
  ```python
  import pandas as pd
  audio_df = pd.read_csv("audio/output/audio_results.csv")
  pdf_df   = pd.read_csv("pdf/output/pdf_results.csv")
  image_df = pd.read_csv("images/output/image_results.csv")
  video_df = pd.read_csv("video/output/video_results.csv")
  text_df  = pd.read_csv("text/output/text_results.csv")

  merged = audio_df.merge(pdf_df, on="Incident_ID", how="outer")
  merged = merged.merge(image_df, on="Incident_ID", how="outer")
  merged = merged.merge(video_df, on="Incident_ID", how="outer")
  merged = merged.merge(text_df, on="Incident_ID", how="outer")
  ```

### Step 3: Handle Missing Values (Day 10)
- [ ] Fill NaN with `"N/A"` or `"No data from this source"`
- [ ] Validate: each row should have data from at least one source

### Step 4: Severity Classification (Day 10-11)
- [ ] Combine severity signals from all modalities
- [ ] Use a rule-based or weighted approach:
  ```python
  def compute_severity(row):
      scores = [row.get('Urgency_Score', 0),
                row.get('Confidence', 0),
                severity_map.get(row.get('Severity_Label', 'Low'), 0)]
      avg = sum(scores) / len(scores)
      if avg > 0.7: return "High"
      elif avg > 0.4: return "Medium"
      else: return "Low"
  ```
- [ ] Generate final column: `Overall_Severity`

### Step 5: Save Final Dataset (Day 11)
- [ ] Output: `/integration/output/final_incident_report.csv`
- [ ] Final schema:
  | Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
  |-------------|------------|-------------|---------------|-------------|-----------------|----------|

---

## 📊 Phase 4 — Dashboard, Documentation & Report (Days 12–13)

### Dashboard (Day 12) — **Team Lead + 1 volunteer**
- [ ] Build dashboard using **Streamlit** (recommended) or Dash
  ```python
  import streamlit as st
  import pandas as pd
  df = pd.read_csv("integration/output/final_incident_report.csv")
  st.title("Multimodal Incident Analyzer")
  st.dataframe(df)
  severity_filter = st.selectbox("Filter by Severity", ["All", "High", "Medium", "Low"])
  ```
- [ ] Features to include:
  - [ ] Filter by severity, source, incident type
  - [ ] Summary statistics (total incidents, severity breakdown)
  - [ ] Individual incident drill-down
  - [ ] Visualizations: bar charts, pie charts for incident distribution

### Pipeline Architecture Diagram (Day 12) — **1 student**
- [ ] Create a visual diagram showing the full data flow:
  ```
  Raw Data → Ingestion → AI Processing → Extraction → Merge → Dashboard
  ```
- [ ] Tools: draw.io, Lucidchart, Mermaid, or PowerPoint
- [ ] Save to `/docs/pipeline_architecture.png`

### Project Report (Days 12–13) — **All students contribute their section**
- [ ] **Section 1:** Problem Understanding & Objectives
- [ ] **Section 2:** Pipeline Architecture explanation
- [ ] **Section 3:** Student 1 — Audio module (approach, tools, challenges, results)
- [ ] **Section 4:** Student 2 — PDF module
- [ ] **Section 5:** Student 3 — Image module
- [ ] **Section 6:** Student 4 — Video module
- [ ] **Section 7:** Student 5 — Text module
- [ ] **Section 8:** Integration & Merging process
- [ ] **Section 9:** Dashboard and querying
- [ ] **Section 10:** Challenges faced & lessons learned
- [ ] Save to `/docs/project_report.pdf`

### README.md (Day 13)
- [ ] Project title and description
- [ ] Team members and roles
- [ ] Installation instructions
- [ ] How to run each module
- [ ] How to run the full pipeline
- [ ] How to launch the dashboard
- [ ] Dataset sources and links

---

## 🎤 Phase 5 — Final Demo & Submission (Day 14)

- [ ] **Prepare demo script:** show raw data → AI processing → structured output → dashboard
- [ ] **Each student demos their module** (2 min each)
- [ ] **Show integration pipeline running** (merge step)
- [ ] **Show dashboard with filters** and incident drill-down
- [ ] **Final code cleanup:** remove debug prints, add docstrings
- [ ] **Push everything to GitHub**
- [ ] **Submit:** repo link, report PDF, demo video (if required)

---

## 📋 Student Assignment Summary

| Student | Module | Dataset Source | Key Tools | Output File |
|---------|--------|---------------|-----------|-------------|
| **S1** | Audio | Kaggle 911 Calls | Whisper, spaCy, HuggingFace | `audio_results.csv` |
| **S2** | PDF | MuckRock FOIA PDFs | pdfplumber, pytesseract, spaCy | `pdf_results.csv` |
| **S3** | Image | Roboflow Fire Detection | YOLOv8, OpenCV, pytesseract | `image_results.csv` |
| **S4** | Video | CAVIAR CCTV Dataset | OpenCV, YOLOv8, moviepy | `video_results.csv` |
| **S5** | Text | Kaggle CrimeReport | spaCy, HuggingFace, NLTK | `text_results.csv` |

---

## ⚠️ Key Deadlines Checklist

| Day | Milestone |
|-----|-----------|
| Day 2 | Repo setup, datasets downloaded, schema agreed |
| Day 6 | Individual AI processing code working |
| Day 8 | Individual CSVs ready with `Incident_ID` |
| Day 11 | Merged dataset complete |
| Day 13 | Dashboard + Report + README done |
| Day 14 | Demo day / Submission |
