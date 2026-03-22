# 🎙️ Audio Module — Student 1 (Audio Analyst)

## Dataset: 911 Recordings (Kaggle)

This module uses the **911 Recordings: The First 6 Seconds** dataset from Kaggle.

### How to Get the Dataset

#### Option A — Use the Kaggle Notebook (Recommended)

The Kaggle notebook already includes working transcription code. You run it on Kaggle and download the output.

1. **Open the notebook:** [911 Calls Wav2Vec2](https://www.kaggle.com/code/stpeteishii/911-calls-wav2vec2)
2. **Sign in** to your Kaggle account
3. Click **"Copy and Edit"** to fork the notebook
4. **Run all cells** (uses free Kaggle GPU)
5. The notebook will produce transcribed text output
6. **Export the results as CSV** and save it to `audio/data/kaggle_transcripts.csv`

Then run the pipeline:
```bash
python audio/audio_analyzer.py
```
The pipeline will auto-detect the CSV and skip Whisper transcription (Mode B).

#### Option B — Download Raw Audio Files

1. Go to the dataset: [911 Recordings: The First 6 Seconds](https://www.kaggle.com/datasets/louisedavis/911-recordings-the-first-6-seconds)
2. Click **Download** and extract the `.wav` files
3. Place all `.wav` files into `audio/data/`

Then run the pipeline:
```bash
python audio/audio_analyzer.py
```
The pipeline will run Whisper speech-to-text on each file (Mode A).

#### Option C — Use Kaggle API (Programmatic)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key: https://www.kaggle.com/docs/api
# Download the dataset
kaggle datasets download -d louisedavis/911-recordings-the-first-6-seconds -p audio/data/ --unzip
```

#### Option D — Quick Test with Sample Files

```bash
python audio/generate_samples.py   # Generates 5 synthetic 911 call audio files
python audio/audio_analyzer.py     # Runs the pipeline on them
```

---

## Data Modes

The pipeline automatically detects which mode to use based on what's in `audio/data/`:

| Mode | Trigger | What Happens |
|------|---------|-------------|
| **Mode A** (Audio) | `.wav`/`.mp3` files found | Whisper transcription → Sentiment → NER → CSV |
| **Mode B** (CSV) | `.csv` file found | Loads transcripts → Sentiment → NER → CSV |

> **Note:** If both `.csv` and audio files exist, Mode B (CSV) takes priority.

---

## Expected CSV Input Format (Mode B)

The pipeline auto-detects column names. Any of these will work:

| Purpose | Accepted Column Names |
|---------|----------------------|
| Transcript text | `transcript`, `transcription`, `text`, `description` |
| Date/time | `date`, `Date`, `timestamp` |
| Location/state | `state`, `State`, `location`, `Location`, `city` |
| Fatalities | `deaths`, `Deaths`, `fatalities` |

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
