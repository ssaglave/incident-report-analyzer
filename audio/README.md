# 🎙️ Audio Module — Student 1 (Audio Analyst)

## Dataset: 911 Recordings (Kaggle)

This module uses the **911 Recordings: The First 6 Seconds** dataset from Kaggle.
- **707 real 911 emergency call audio files** (.wav, 6 seconds each)
- **Metadata CSV** with title, date, state, deaths, description for each call

### How to Get the Dataset

#### Option A — Kaggle API (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key: https://www.kaggle.com/docs/api
# Download and extract the dataset
kaggle datasets download -d louisteitelbaum/911-recordings-first-6-seconds -p audio/data/ --unzip
```

This places `911_first6sec/` folder with 707 `.wav` files + `911_metadata.csv` into `audio/data/`.

Then run the pipeline:
```bash
python audio/audio_analyzer.py          # Process ALL 707 calls
python audio/audio_analyzer.py --max 10 # Process first 10 only (quick test)
```

#### Option B — Manual Download from Kaggle

1. Go to: [911 Recordings: The First 6 Seconds](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds)
2. **Sign in** to your Kaggle account
3. Click **Download** and extract the zip
4. Place the extracted `911_first6sec/` folder into `audio/data/`

#### Option C — Use the Kaggle Notebook (Pre-transcribed)

1. **Open the notebook:** [911 Calls Wav2Vec2](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2)
2. Click **"Copy and Edit"** to fork and run it on Kaggle
3. Export the transcribed results as CSV → save to `audio/data/transcripts.csv`
4. The pipeline will auto-detect the CSV (Mode B) and skip Whisper transcription

#### Option D — Quick Test with Synthetic Samples

```bash
python audio/generate_samples.py   # Generates 5 synthetic 911 call audio files
python audio/audio_analyzer.py     # Runs the pipeline on them
```

---

## Data Modes

The pipeline auto-detects which mode to use based on what's in `audio/data/`:

| Mode | Trigger | What Happens |
|------|---------|-------------|
| **Mode A** (Kaggle) | `911_metadata.csv` found | Load metadata + Whisper on .wav → Sentiment → NER → CSV |
| **Mode B** (CSV) | Other `.csv` with transcript column | Load transcripts → Sentiment → NER → CSV |
| **Mode C** (Audio) | Only `.wav`/`.mp3` files | Whisper transcription → Sentiment → NER → CSV |

> **Note:** Mode A (Kaggle) is checked first, then Mode B, then Mode C.

---

## Kaggle Metadata Columns Used

The `911_metadata.csv` includes these columns — all used by the pipeline:

| Column | How It's Used |
|--------|-------------|
| `filename` | Links to the .wav audio file for Whisper transcription |
| `description` | Combined with Whisper transcript for richer event classification |
| `title` | Used as supplementary event info |
| `date` | Mapped to Timestamp in output |
| `state` | Used as fallback Location if NER finds none |
| `deaths` | Boosts urgency score when > 0 |
| `potential_death` | Slightly boosts urgency score |

---

## Output

The pipeline produces `audio/output/audio_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `Incident_ID` | Shared key (INC_001, INC_002, ...) |
| `Source` | Always "audio" |
| `Call_ID` | Unique call identifier (C001, C002, ...) |
| `Transcript` | Full transcribed text |
| `Extracted_Event` | Classified event type |
| `Location` | Extracted location(s) from NER |
| `Timestamp` | Date from CSV metadata (if available) |
| `Sentiment` | Distressed / Calm |
| `Urgency_Score` | 0.00 – 1.00 |
| `Severity` | High / Medium / Low |

---

## AI Tools Used

| Tool | Purpose |
|------|---------|
| **OpenAI Whisper** (`base` model) | Speech-to-text transcription |
| **HuggingFace** (`distilbert-base-uncased-finetuned-sst-2-english`) | Sentiment analysis |
| **spaCy** (`en_core_web_sm`) | Named Entity Recognition |
| Custom keyword matching | Event classification |
| Weighted scoring | Urgency score calculation |

## Prerequisites

```bash
pip install openai-whisper spacy transformers torch pandas
python -m spacy download en_core_web_sm
brew install ffmpeg   # Required by Whisper for audio decoding
```
