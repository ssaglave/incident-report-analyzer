# 🚨 Multimodal Crime / Incident Report Analyzer

An AI-powered system that automatically processes multiple types of unstructured data (audio, PDFs, images, video, text) and converts them into a unified, structured incident report for emergency response teams.

---

## 📁 Project Structure

```
incident-report-analyzer/
├── audio/                  # Student 1 — Audio Analyst
│   ├── data/               # Raw audio files (911 calls)
│   └── output/             # Processed CSV output
├── pdf/                    # Student 2 — Document Analyst
│   ├── data/               # Raw PDF police reports
│   └── output/             # Processed CSV output
├── images/                 # Student 3 — Image Analyst
│   ├── data/               # Raw scene photographs
│   └── output/             # Processed CSV + annotated images
│       └── annotated/      # Images with detection overlays
├── video/                  # Student 4 — Video Analyst
│   ├── data/               # Raw CCTV video clips
│   └── output/             # Processed CSV + extracted frames
│       └── frames/         # Key frames extracted from video
├── text/                   # Student 5 — Text Analyst
│   ├── data/               # Raw text data (crime reports)
│   └── output/             # Processed CSV output
├── integration/            # Final pipeline merging all outputs
│   └── output/             # Merged incident report CSV
├── dashboard/              # Streamlit dashboard for visualization
├── docs/                   # Reports, diagrams, documentation
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ssaglave/incident-report-analyzer.git
cd incident-report-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

---

## 🚀 How to Run

### Individual Modules

```bash
# Student 1 — Audio Analysis
python audio/audio_analyzer.py

# Student 2 — PDF Analysis
python pdf/pdf_analyzer.py

# Student 3 — Image Analysis
python images/image_analyzer.py

# Student 4 — Video Analysis
python video/video_analyzer.py

# Student 5 — Text Analysis
python text/text_analyzer.py
```

### Integration Pipeline

```bash
# Merge all outputs into final incident report
python integration/merge_pipeline.py
```

### Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

---

## 📊 Final Output

The integrated pipeline produces a unified CSV at `integration/output/final_incident_report.csv` with the following structure:

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|------------|-------------|---------------|-------------|-----------------|----------|

---

## 👥 Team Members

| Student | Role | Module |
|---------|------|--------|
| Student 1 | Audio Analyst | `/audio` |
| Student 2 | Document Analyst | `/pdf` |
| Student 3 | Image Analyst | `/images` |
| Student 4 | Video Analyst | `/video` |
| Student 5 | Text Analyst | `/text` |

---

## 📦 Datasets

| Module | Dataset | Source |
|--------|---------|--------|
| Audio | 911 Calls + Wav2Vec2 | [Kaggle](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2) |
| PDF | Arkansas PD 1033 Training Proposals | [MuckRock](https://www.muckrock.com) |
| Image | Fire Detection Dataset | [Roboflow](https://universe.roboflow.com) |
| Video | CAVIAR CCTV Dataset | [University of Edinburgh](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) |
| Text | CrimeReport Dataset | [Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport) |

---

## 📄 License

This project is developed as part of the **AI for Engineers** course assignment.
