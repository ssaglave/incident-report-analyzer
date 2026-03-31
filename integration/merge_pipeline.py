"""
Final Integration Script — merge_pipeline.py
Merges all 5 module outputs (audio, pdf, image, video, text)
into a single unified incident report dataset.

Place this file in: /integration/merge_pipeline.py

Usage:
    python integration/merge_pipeline.py
    python integration/merge_pipeline.py --output integration/output/final_incident_report.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────────
# Paths are relative to project root (parent of integration/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODULE_OUTPUTS = {
    "audio": os.path.join(PROJECT_ROOT, "audio", "output", "audio_results.csv"),
    "pdf":   os.path.join(PROJECT_ROOT, "pdf", "output", "pdf_results.csv"),
    "image": os.path.join(PROJECT_ROOT, "images", "output", "image_results.csv"),
    "video": os.path.join(PROJECT_ROOT, "video", "output", "video_results.csv"),
    "text":  os.path.join(PROJECT_ROOT, "text", "output", "text_results.csv"),
}

DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "integration", "output", "final_incident_report.csv"
)

# Column mapping: module column → unified column name
# Each module's key columns are mapped to the unified schema
COLUMN_MAPPING = {
    "audio": {
        "event_col":    "Extracted_Event",      # → Audio_Event
        "severity_col": "Urgency_Score",        # numeric 0–1
        "extra_cols":   ["Transcript", "Location", "Sentiment"],
    },
    "pdf": {
        "event_col":    "Doc_Type",             # → PDF_Doc_Type
        "severity_col": "Severity",             # Low/Medium/High
        "extra_cols":   ["Department", "Date", "Location", "Officer", "Summary"],
    },
    "image": {
        "event_col":    "Objects_Detected",     # → Image_Objects
        "severity_col": "Confidence",           # numeric 0–1
        "extra_cols":   ["Scene_Type", "Bounding_Boxes"],
    },
    "video": {
        "event_col":    "Video_Event",          # → Video_Event
        "severity_col": "Severity",             # Low/Medium/High/Critical
        "extra_cols":   ["Detected_Objects", "Motion_Score", "Confidence"],
    },
    "text": {
        "event_col":    "Crime_Type",           # → Text_Crime_Type
        "severity_col": "Severity",             # Low/Medium/High
        "extra_cols":   ["Location_Entity", "Sentiment", "Topic"],
    },
}

# Unified column names for the final output
UNIFIED_COLUMNS = {
    "audio": "Audio_Event",
    "pdf":   "PDF_Doc_Type",
    "image": "Image_Objects",
    "video": "Video_Event",
    "text":  "Text_Crime_Type",
}

SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
RANK_TO_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}


# ── Step 1: Load Module Outputs ───────────────────────────────────────────────

def load_module_csv(module_name: str) -> pd.DataFrame:
    """Load a module's output CSV, returning an empty DataFrame if missing."""
    path = MODULE_OUTPUTS[module_name]
    if not os.path.exists(path):
        print(f"  ⚠  {module_name.upper():6s} — file not found: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        print(f"  ✓  {module_name.upper():6s} — {len(df)} records loaded from {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"  ✗  {module_name.upper():6s} — error reading: {e}")
        return pd.DataFrame()


def load_all_modules() -> dict:
    """Load all 5 module outputs."""
    print("\n" + "=" * 65)
    print("  STEP 1: Loading Module Outputs")
    print("=" * 65)

    modules = {}
    for name in MODULE_OUTPUTS:
        modules[name] = load_module_csv(name)
    return modules


# ── Step 2: Define Common Incident_ID & Merge ─────────────────────────────────

def normalize_incident_ids(modules: dict) -> dict:
    """
    Ensure all modules use consistent Incident_ID format (INC_001, INC_002, ...).

    Each module generates IDs independently (INC_001, INC_002, ...) during its
    own processing.  For the integration step, matching IDs across modules links
    records that conceptually belong to the same incident.

    Where a module has more records than another, the extra records will be
    retained and the missing modality columns will be filled with 'N/A'.
    """
    print("\n" + "=" * 65)
    print("  STEP 2: Normalizing Incident IDs & Merging")
    print("=" * 65)

    for name, df in modules.items():
        if df.empty:
            continue

        # If the module already has Incident_ID, standardize the format
        if "Incident_ID" in df.columns:
            # Ensure consistent formatting: INC_001, INC_002, etc.
            df["Incident_ID"] = df["Incident_ID"].astype(str).str.strip()
            print(f"  ✓  {name.upper():6s} — {df['Incident_ID'].nunique()} unique Incident_IDs")
        else:
            # Generate Incident_IDs if missing
            df["Incident_ID"] = [f"INC_{i:03d}" for i in range(1, len(df) + 1)]
            print(f"  ✓  {name.upper():6s} — assigned {len(df)} Incident_IDs")

        modules[name] = df

    return modules


def extract_unified_columns(modules: dict) -> dict:
    """
    Extract and rename each module's key column to the unified schema name.
    Returns a dict of DataFrames with only Incident_ID + unified column.
    """
    unified_dfs = {}

    for name, df in modules.items():
        if df.empty:
            unified_col = UNIFIED_COLUMNS[name]
            unified_dfs[name] = pd.DataFrame(
                columns=["Incident_ID", unified_col, f"{name}_Severity"]
            )
            continue

        mapping = COLUMN_MAPPING[name]
        unified_col = UNIFIED_COLUMNS[name]

        # Build the module's contribution to the unified dataset
        result = pd.DataFrame()
        result["Incident_ID"] = df["Incident_ID"]

        # Map the event/type column to the unified name
        if mapping["event_col"] in df.columns:
            result[unified_col] = df[mapping["event_col"]].astype(str)
        else:
            result[unified_col] = "N/A"

        # Carry over the severity signal (keep original name for later combination)
        if mapping["severity_col"] in df.columns:
            result[f"{name}_Severity"] = df[mapping["severity_col"]]
        else:
            result[f"{name}_Severity"] = "N/A"

        unified_dfs[name] = result

    return unified_dfs


def merge_all(unified_dfs: dict) -> pd.DataFrame:
    """Merge all module DataFrames on Incident_ID using outer join."""

    # Start with the first non-empty DataFrame
    non_empty = [(name, df) for name, df in unified_dfs.items() if not df.empty]

    if not non_empty:
        print("  ✗  No data to merge!")
        return pd.DataFrame()

    merged = non_empty[0][1]
    print(f"  ✓  Starting merge with {non_empty[0][0].upper()} ({len(merged)} rows)")

    for name, df in non_empty[1:]:
        merged = merged.merge(df, on="Incident_ID", how="outer")
        print(f"  ✓  Merged {name.upper():6s} → {len(merged)} rows total")

    return merged


# ── Step 3: Handle Missing Values ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values where a modality has no data for a given incident."""
    print("\n" + "=" * 65)
    print("  STEP 3: Handling Missing Values")
    print("=" * 65)

    # Count missing before
    missing_before = df.isna().sum().sum()

    # Fill missing values with 'N/A'
    unified_cols = list(UNIFIED_COLUMNS.values())
    for col in unified_cols:
        if col in df.columns:
            df[col] = df[col].fillna("N/A").replace("", "N/A").replace("nan", "N/A")

    # Fill severity columns with 'N/A'
    severity_cols = [c for c in df.columns if c.endswith("_Severity")]
    for col in severity_cols:
        df[col] = df[col].fillna("N/A").replace("", "N/A").replace("nan", "N/A")

    missing_after = df.isna().sum().sum()

    print(f"  ✓  Filled {missing_before - missing_after} missing values with 'N/A'")
    print(f"  ✓  Remaining NaN: {missing_after}")

    return df


# ── Step 4: Generate Final Severity Classification ────────────────────────────

def _parse_severity(value) -> int:
    """Convert a severity value to a numeric rank (1–4)."""
    if pd.isna(value) or str(value).strip() in ("N/A", "", "nan"):
        return 0

    val_str = str(value).strip()

    # Check if it's a text label
    if val_str in SEVERITY_RANK:
        return SEVERITY_RANK[val_str]

    # Check if it's a numeric score (0–1 range, e.g., urgency or confidence)
    try:
        score = float(val_str)
        if 0.0 <= score <= 1.0:
            if score >= 0.75:
                return SEVERITY_RANK["High"]
            elif score >= 0.45:
                return SEVERITY_RANK["Medium"]
            else:
                return SEVERITY_RANK["Low"]
        return 0
    except (ValueError, TypeError):
        return 0


def compute_overall_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a final severity classification (Low / Medium / High / Critical)
    based on combined signals from all modalities.

    Strategy:
    - Parse each module's severity signal to a numeric rank
    - Take the maximum severity across all available modalities
    - If multiple modules agree on High/Critical, escalate
    """
    print("\n" + "=" * 65)
    print("  STEP 4: Computing Overall Severity")
    print("=" * 65)

    severity_cols = [c for c in df.columns if c.endswith("_Severity")]

    severity_ranks = pd.DataFrame()
    for col in severity_cols:
        severity_ranks[col] = df[col].apply(_parse_severity)

    # Maximum severity across all modules
    df["Max_Severity_Rank"] = severity_ranks.max(axis=1)

    # Count how many modules report High (3) or Critical (4)
    high_critical_count = (severity_ranks >= 3).sum(axis=1)

    # Count how many modules have valid data (rank > 0)
    valid_count = (severity_ranks > 0).sum(axis=1)

    # Escalation rule: if 2+ modules report High/Critical, bump up by 1 rank
    escalated_rank = df["Max_Severity_Rank"].copy()
    escalation_mask = high_critical_count >= 2
    escalated_rank[escalation_mask] = np.minimum(
        escalated_rank[escalation_mask] + 1, 4  # cap at Critical
    )

    # Map back to label
    df["Severity"] = escalated_rank.map(RANK_TO_SEVERITY).fillna("Low")

    # Clean up temporary column
    df.drop(columns=["Max_Severity_Rank"], inplace=True)

    # Print severity distribution
    severity_counts = df["Severity"].value_counts()
    print(f"  ✓  Severity distribution:")
    for sev in ["Critical", "High", "Medium", "Low"]:
        count = severity_counts.get(sev, 0)
        if count > 0:
            print(f"      {sev:10s}: {count}")

    return df


# ── Step 5: Finalize Output ───────────────────────────────────────────────────

def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arrange columns in the required unified schema order and clean up.

    Final structure:
    Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity
    """
    print("\n" + "=" * 65)
    print("  STEP 5: Finalizing Unified Dataset")
    print("=" * 65)

    # Define the required output column order
    output_columns = [
        "Incident_ID",
        "Audio_Event",
        "PDF_Doc_Type",
        "Image_Objects",
        "Video_Event",
        "Text_Crime_Type",
        "Severity",
    ]

    # Ensure all required columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = "N/A"

    # Select and reorder
    final_df = df[output_columns].copy()

    # Sort by Incident_ID
    final_df = final_df.sort_values("Incident_ID").reset_index(drop=True)

    # Final cleanup: replace any remaining empty strings or NaN
    final_df = final_df.fillna("N/A").replace("", "N/A")

    print(f"  ✓  Final dataset: {len(final_df)} rows × {len(final_df.columns)} columns")
    print(f"  ✓  Columns: {list(final_df.columns)}")

    return final_df


def save_output(df: pd.DataFrame, output_path: str) -> str:
    """Save the final unified dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n  💾 Saved: {output_path} ({len(df)} rows)")
    return output_path


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_integration(output_path: str = None) -> pd.DataFrame:
    """
    Run the full integration pipeline:
      Step 1 → Load all module outputs
      Step 2 → Normalize IDs & merge on Incident_ID
      Step 3 → Handle missing values
      Step 4 → Compute overall severity
      Step 5 → Finalize & save unified dataset
    """
    output_path = output_path or DEFAULT_OUTPUT

    print("\n" + "=" * 65)
    print("  🔗 FINAL INTEGRATION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Step 1: Load
    modules = load_all_modules()

    loaded_count = sum(1 for df in modules.values() if not df.empty)
    if loaded_count == 0:
        print("\n  ✗ No module outputs found. Run individual modules first.")
        print("    Example: python run_pipeline.py")
        return pd.DataFrame()

    print(f"\n  📊 Loaded {loaded_count}/5 module outputs")

    # Step 2: Normalize IDs, extract unified columns, merge
    modules = normalize_incident_ids(modules)
    unified_dfs = extract_unified_columns(modules)
    merged = merge_all(unified_dfs)

    if merged.empty:
        print("\n  ✗ Merge produced empty dataset.")
        return pd.DataFrame()

    # Step 3: Handle missing values
    merged = handle_missing_values(merged)

    # Step 4: Compute overall severity
    merged = compute_overall_severity(merged)

    # Step 5: Finalize and save
    final = finalize_output(merged)
    save_output(final, output_path)

    # Summary
    print("\n" + "=" * 65)
    print("  ✅ INTEGRATION COMPLETE")
    print("=" * 65)
    print(f"\n  Preview (first 5 rows):")
    print(final.head().to_string(index=False))
    print()

    return final


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge all 5 module outputs into a unified incident report."
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help="Path for the final merged CSV (default: integration/output/final_incident_report.csv)",
    )
    args = parser.parse_args()

    result = run_integration(output_path=args.output)

    if result.empty:
        sys.exit(1)
