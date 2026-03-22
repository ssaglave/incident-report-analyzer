# 📋 Common Output Schema — Multimodal Incident Report Analyzer

> **Every student MUST follow this schema** to ensure all five outputs merge cleanly in the integration phase.

---

## 🔑 Shared Rules

1. **Every CSV must include an `Incident_ID` column** — this is the merge key.
2. **Format:** `INC_001`, `INC_002`, etc. (zero-padded, 3 digits minimum).
3. **Every CSV must include a `Source` column** with the modality name.
4. **Timestamps** must use ISO 8601: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-04-10 14:32:00`).
5. **Severity** must be one of: `Low`, `Medium`, `High`.
6. **Confidence scores** must be a float between `0.00` and `1.00`.
7. **Missing values:** Use empty string `""` — do NOT use `N/A`, `None`, or `null`.
8. **Encoding:** UTF-8.
9. **File format:** `.csv` with comma delimiter.

---

## 📊 Individual Module Schemas

### Student 1 — Audio (`audio/output/audio_results.csv`)

| Column           | Type    | Description                                  | Example                                  |
|------------------|---------|----------------------------------------------|------------------------------------------|
| `Incident_ID`    | string  | Shared incident key                          | `INC_001`                                |
| `Source`          | string  | Always `"audio"`                             | `audio`                                  |
| `Call_ID`        | string  | Unique call identifier                       | `C001`                                   |
| `Transcript`     | string  | Full or excerpt of transcribed text          | `There is a fire, people are trapped...` |
| `Extracted_Event`| string  | Detected incident type                       | `Building fire / trapped persons`        |
| `Location`       | string  | Location mentioned in audio                  | `Downtown Ave`                           |
| `Timestamp`      | string  | Date/time of call (ISO 8601)                 | `2025-04-10 14:32:00`                    |
| `Sentiment`      | string  | Detected sentiment                           | `Distressed`                             |
| `Urgency_Score`  | float   | Urgency level (0.00–1.00)                    | `0.91`                                   |
| `Severity`       | string  | Low / Medium / High                          | `High`                                   |

---

### Student 2 — PDF (`pdf/output/pdf_results.csv`)

| Column           | Type    | Description                                  | Example                                  |
|------------------|---------|----------------------------------------------|------------------------------------------|
| `Incident_ID`    | string  | Shared incident key                          | `INC_001`                                |
| `Source`          | string  | Always `"pdf"`                               | `pdf`                                    |
| `Report_ID`      | string  | Unique report identifier                     | `RPT_001`                                |
| `Department`     | string  | Department / agency name                     | `Arkansas PD`                            |
| `Doc_Type`       | string  | Document category                            | `1033 Training Proposal`                 |
| `Date`           | string  | Report date (ISO 8601)                       | `2015-04-10`                             |
| `Location`       | string  | Location mentioned in report                 | `Little Rock, AR`                        |
| `Officer`        | string  | Officer or author name                       | `Sgt. Johnson`                           |
| `Summary`        | string  | Key extracted details                        | `Equipment request: tactical gear`       |
| `Severity`       | string  | Low / Medium / High                          | `Medium`                                 |

---

### Student 3 — Image (`images/output/image_results.csv`)

| Column             | Type    | Description                                  | Example                                |
|--------------------|---------|----------------------------------------------|----------------------------------------|
| `Incident_ID`      | string  | Shared incident key                          | `INC_001`                              |
| `Source`            | string  | Always `"image"`                             | `image`                                |
| `Image_ID`         | string  | Unique image identifier                      | `IMG_034`                              |
| `Scene_Type`       | string  | Classified scene category                    | `Fire Scene`                           |
| `Objects_Detected` | string  | Comma-separated detected objects             | `fire, smoke, person`                  |
| `Bounding_Boxes`   | string  | Summary of detection regions                 | `2 fire regions, 1 smoke plume`        |
| `Text_Extracted`   | string  | OCR text from image                          | `License: ABC 1234`                    |
| `Confidence`       | float   | Detection confidence (0.00–1.00)             | `0.94`                                 |
| `Severity`         | string  | Low / Medium / High                          | `High`                                 |

---

### Student 4 — Video (`video/output/video_results.csv`)

| Column           | Type    | Description                                  | Example                                |
|------------------|---------|----------------------------------------------|----------------------------------------|
| `Incident_ID`    | string  | Shared incident key                          | `INC_001`                              |
| `Source`          | string  | Always `"video"`                             | `video`                                |
| `Clip_ID`        | string  | Unique clip identifier                       | `CAVIAR_03`                            |
| `Timestamp`      | string  | Time position in clip                        | `00:00:12`                             |
| `Frame_ID`       | string  | Specific frame reference                     | `FRM_036`                              |
| `Event_Detected` | string  | What happened in the frame                   | `Person collapsing`                    |
| `Persons_Count`  | integer | Number of people detected                    | `1`                                    |
| `Confidence`     | float   | Detection confidence (0.00–1.00)             | `0.88`                                 |
| `Severity`       | string  | Low / Medium / High                          | `High`                                 |

---

### Student 5 — Text (`text/output/text_results.csv`)

| Column            | Type    | Description                                  | Example                                |
|-------------------|---------|----------------------------------------------|----------------------------------------|
| `Incident_ID`     | string  | Shared incident key                          | `INC_001`                              |
| `Source`           | string  | Always `"text"`                              | `text`                                 |
| `Text_ID`         | string  | Unique text record identifier                | `TXT_112`                              |
| `Raw_Text`        | string  | Original text (truncated if needed)          | `Robbery reported on Oak Street...`    |
| `Crime_Type`      | string  | Classified crime category                    | `Robbery`                              |
| `Location_Entity` | string  | Extracted location                           | `Oak Street, Chicago`                  |
| `Entities`        | string  | Comma-separated NER entities                 | `John Doe (PERSON), Chicago (GPE)`     |
| `Sentiment`       | string  | Sentiment label                              | `Negative`                             |
| `Topic`           | string  | Classified topic                             | `Theft / Robbery`                      |
| `Severity`        | string  | Low / Medium / High                          | `High`                                 |

---

## 🔗 Final Integrated Schema (`integration/output/final_incident_report.csv`)

| Column            | Type    | Comes From      | Description                          |
|-------------------|---------|-----------------|--------------------------------------|
| `Incident_ID`     | string  | All             | Shared merge key                     |
| `Audio_Event`     | string  | Student 1       | Event from audio transcript          |
| `Audio_Urgency`   | float   | Student 1       | Urgency score from audio             |
| `PDF_Doc_Type`    | string  | Student 2       | Document type from PDF               |
| `PDF_Summary`     | string  | Student 2       | Key details from PDF                 |
| `Image_Objects`   | string  | Student 3       | Detected objects in images           |
| `Image_Confidence`| float   | Student 3       | Image detection confidence           |
| `Video_Event`     | string  | Student 4       | Event detected in video              |
| `Video_Confidence`| float   | Student 4       | Video detection confidence           |
| `Text_Crime_Type` | string  | Student 5       | Crime type from text analysis        |
| `Text_Sentiment`  | string  | Student 5       | Sentiment from text                  |
| `Overall_Severity`| string  | Integration     | Combined severity: Low/Medium/High   |

---

## 🏷️ Severity Classification Rules

Use these rules consistently across all modules:

| Severity | Criteria |
|----------|----------|
| **High** | Life-threatening events: fire with trapped persons, active violence, weapon detected, person collapsing, urgency score > 0.7 |
| **Medium** | Property crime, theft, vehicle accident without injuries, moderate distress |
| **Low** | Noise complaints, minor disturbances, routine reports, low urgency |

---

## ✅ Checklist Before Submitting Your CSV

- [ ] `Incident_ID` column exists and follows `INC_XXX` format
- [ ] `Source` column is filled with your modality name
- [ ] `Severity` column uses only `Low`, `Medium`, or `High`
- [ ] Confidence scores are between `0.00` and `1.00`
- [ ] No `N/A` or `null` — use empty string `""` for missing values
- [ ] File is saved as UTF-8 CSV with comma delimiter
- [ ] File is in your module's `output/` folder
