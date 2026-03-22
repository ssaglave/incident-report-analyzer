# 🚨 Multimodal Crime / Incident Report Analyzer — Task Plan

## Project Timeline: 1 Week (7 Days)

---

## 📅 Phase Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1** | Day 1 | Setup, Dataset Collection, Environment Config |
| **Phase 2** | Days 2–4 | Individual Module Development (per student) |
| **Phase 3** | Day 5 | Integration & Unified Pipeline |
| **Phase 4** | Day 6 | Dashboard, Documentation, Report |
| **Phase 5** | Day 7 | Final Demo Preparation & Submission |

---

## 🔧 Phase 1 — Project Setup (Day 1) — **All Students**

> [!IMPORTANT]
> Every student must complete these setup tasks before starting individual work.

- [x] Create a shared **GitHub repository** with folder structure:
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
- [x] Set up a **shared Python virtual environment** and agree on Python version (3.10+ recommended)
- [x] Create `requirements.txt` with common dependencies
- [x] **Agree on a common output schema** — every student's CSV must include:
  - `Incident_ID` (shared key for merging)
  - `Source` (audio / pdf / image / video / text)
  - `Event` / `Event_Detected`
  - `Location`
  - `Timestamp` or `Date`
  - `Severity` (Low / Medium / High)
- [x] Each student **downloads their dataset** (links in assignment brief)
- [x] Create the **AI Pipeline Architecture Diagram**

---

## 👤 Phase 2 — Individual Module Development (Days 2–4)

Each student works independently on their module. Every module follows the same 3-stage pattern:

```
Stage A: Data Ingestion & Preprocessing
Stage B: AI Model Processing & Extraction
Stage C: Structured Output Generation (CSV)
```

---

### 🎙️ Student 1 — Audio Analyst

**Goal:** Convert emergency audio calls → structured CSV

#### Stage A: Data Ingestion (Day 2 — Morning)
- [x] Download 911 Calls dataset from Kaggle ([link](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds))
- [x] Load audio files into processing script; handle file formats (.wav, .mp3)

#### Stage B: AI Processing (Day 2 Afternoon – Day 3)
- [x] **Speech-to-Text:** Use `openai-whisper` to transcribe audio files
  ```python
  import whisper
  model = whisper.load_model("base")
  result = model.transcribe("audio_file.wav")
  transcript = result["text"]
  ```
- [x] **Keyword Extraction:** Use `spaCy` NER to extract:
  - Incident type (fire, accident, robbery, etc.)
  - Location mentions (street names, landmarks)
  - Person names
  - Urgency phrases ("help!", "trapped", "hurry")
- [x] **Sentiment / Urgency Analysis:** Use HuggingFace `transformers`
  ```python
  from transformers import pipeline
  sentiment = pipeline("sentiment-analysis")
  result = sentiment(transcript)
  ```
- [x] Calculate an **Urgency Score** (0.0–1.0) based on sentiment + keyword presence

#### Stage C: Structured Output (Day 4 — Morning)
- [x] Generate output CSV with columns:
  | Call_ID | Transcript | Extracted_Event | Location | Sentiment | Urgency_Score |
  |---------|-----------|-----------------|----------|-----------|---------------|
- [x] Add `Incident_ID` column for integration
- [x] Validate output: check for nulls, normalize text fields
- [x] Save to `/audio/output/audio_results.csv`

---

### 📄 Student 2 — Document Analyst

**Goal:** Extract structured data from police report PDFs → structured CSV

#### Stage A: Data Ingestion (Day 2 — Morning)
- [ ] Download Arkansas Police Department 1033 Training PDF from [MuckRock](https://www.muckrock.com)
- [ ] Collect 2–3 additional sample police report PDFs (public FOIA docs)
- [ ] Categorize PDFs: text-based vs. scanned (image-based)

#### Stage B: AI Processing (Day 2 Afternoon – Day 3)
- [ ] **Text Extraction (text-based PDFs):** Use `pdfplumber`
- [ ] **OCR (scanned PDFs):** Use `pytesseract`
- [ ] **Named Entity Recognition:** Use `spaCy` to extract:
  - Department name, document type, date, officer names, suspect descriptions
- [ ] **Table Extraction:** Use `pdfplumber` to extract embedded tables

#### Stage C: Structured Output (Day 4 — Morning)
- [ ] Generate output CSV with columns:
  | Report_ID | Department | Doc_Type | Date | Program | Key_Detail |
  |-----------|-----------|----------|------|---------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save to `/pdf/output/pdf_results.csv`

---

### 🖼️ Student 3 — Image Analyst

**Goal:** Detect objects and classify scenes in incident photos → structured CSV

#### Stage A: Data Ingestion (Day 2 — Morning)
- [ ] Download fire detection dataset from [Roboflow](https://universe.roboflow.com) in YOLOv8 format
- [ ] Choose a dataset with **1000+ images** and a trained model badge
- [ ] Organize images into `/images/data/`

#### Stage B: AI Processing (Day 2 Afternoon – Day 3)
- [ ] **Object Detection:** Use `YOLOv8` (ultralytics)
- [ ] Extract: detected objects, bounding boxes, confidence scores
- [ ] **Scene Classification:** Use a pre-trained classifier (ResNet / ViT)
- [ ] **OCR on images:** Use `pytesseract` for text in images

#### Stage C: Structured Output (Day 4 — Morning)
- [ ] Generate output CSV with columns:
  | Image_ID | Scene_Type | Objects_Detected | Bounding_Boxes | Confidence |
  |----------|-----------|------------------|----------------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save to `/images/output/image_results.csv`

---

### 🎥 Student 4 — Video Analyst

**Goal:** Process CCTV footage to detect events → structured CSV

#### Stage A: Data Ingestion (Day 2 — Morning)
- [ ] Download 3–5 short clips from [CAVIAR Dataset](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- [ ] Recommended folders: `Browse`, `OneStopEnter`, `Fight`, `Collapse`
- [ ] Convert `.mpg` to `.mp4` if needed

#### Stage B: AI Processing (Day 2 Afternoon – Day 3)
- [ ] **Frame Extraction:** Use `OpenCV` to extract frames at intervals
- [ ] **Motion Detection:** Frame differencing + background subtraction (MOG2)
- [ ] **Object/Activity Detection:** Run `YOLOv8` on key frames
- [ ] **Anomaly Flagging:** Mark sudden motion spikes, person count changes

#### Stage C: Structured Output (Day 4 — Morning)
- [ ] Generate output CSV with columns:
  | Clip_ID | Timestamp | Frame_ID | Event_Detected | Persons_Count | Confidence |
  |---------|-----------|----------|----------------|---------------|------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save to `/video/output/video_results.csv`

---

### 📝 Student 5 — Text Analyst

**Goal:** Analyze crime text reports using NLP → structured CSV

#### Stage A: Data Ingestion (Day 2 — Morning)
- [ ] Download CrimeReport dataset from [Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport)
- [ ] Load CSV and explore: columns, data types, missing values

#### Stage B: AI Processing (Day 2 Afternoon – Day 3)
- [ ] **Text Preprocessing:** NLTK / spaCy (tokenize, stopwords, lemmatize)
- [ ] **Named Entity Recognition:** Use `spaCy` for persons, locations, orgs, dates
- [ ] **Sentiment Analysis:** Use HuggingFace `transformers`
- [ ] **Topic Classification:** Zero-shot classification
- [ ] **Severity Labeling:** Rule-based severity assignment

#### Stage C: Structured Output (Day 4 — Morning)
- [ ] Generate output CSV with columns:
  | Text_ID | Crime_Type | Location_Entity | Sentiment | Topic | Severity_Label |
  |---------|-----------|-----------------|-----------|-------|----------------|
- [ ] Add `Incident_ID` column for integration
- [ ] Save to `/text/output/text_results.csv`

---

## 🔗 Phase 3 — Integration & Unified Pipeline (Day 5) — **All Students**

> [!IMPORTANT]
> This is the **most critical deliverable**. All 5 outputs merge here.

### Step 1: Define Common Incident_ID (Morning)
- [ ] Finalize Incident_ID format: `INC_001`, `INC_002`, etc.
- [ ] Each student maps their records to shared Incident_IDs

### Step 2: Merge DataFrames (Morning–Afternoon)
- [ ] Write integration script: `/integration/merge_pipeline.py`
- [ ] Merge all 5 output CSVs on `Incident_ID` (outer join)

### Step 3: Handle Missing Values & Severity (Afternoon)
- [ ] Fill NaN with `"N/A"`
- [ ] Combine severity signals from all modalities → `Overall_Severity`
- [ ] Output: `/integration/output/final_incident_report.csv`

---

## 📊 Phase 4 — Dashboard, Documentation & Report (Day 6)

### Dashboard (Morning) — **Team Lead + 1 volunteer**
- [ ] Build dashboard using **Streamlit**
- [ ] Features: filter by severity/source/type, summary stats, drill-down, charts
- [ ] Save to `/dashboard/app.py`

### Pipeline Architecture Diagram (Morning)
- [x] Create visual diagram of full data flow
- [x] Save to `/docs/pipeline_architecture.png`

### Project Report (Afternoon) — **All students contribute their section**
- [ ] **Sections 1–2:** Problem Understanding, Pipeline Architecture
- [ ] **Sections 3–7:** Each student writes their module section (approach, tools, results)
- [ ] **Sections 8–9:** Integration & Dashboard
- [ ] **Section 10:** Challenges & Lessons Learned
- [ ] Save to `/docs/project_report.pdf`

### README.md (Evening)
- [ ] Project description, team roles, install/run instructions, dataset links

---

## 🎤 Phase 5 — Final Demo & Submission (Day 7)

- [ ] **Prepare demo script:** raw data → AI processing → structured output → dashboard
- [ ] **Each student demos their module** (2 min each)
- [ ] **Show integration pipeline + dashboard** with filters
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
| Day 1 | ✅ Repo setup, datasets downloaded, schema agreed |
| Day 3 | Individual AI processing code working |
| Day 4 | Individual CSVs ready with `Incident_ID` |
| Day 5 | Merged dataset complete |
| Day 6 | Dashboard + Report + README done |
| Day 7 | Demo day / Submission |
