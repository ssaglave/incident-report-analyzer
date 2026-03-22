"""
Run Pipeline — Master runner for all 5 modules.

Run all modules:     python run_pipeline.py
Run single module:   python run_pipeline.py audio
Run multiple:        python run_pipeline.py audio text pdf
"""

import sys
import os
import pandas as pd

# Import all 5 student pipelines
from audio.audio_analyzer import AudioPipeline
from pdf.pdf_analyzer import PDFPipeline
from images.image_analyzer import ImagePipeline
from video.video_analyzer import VideoPipeline
from text.text_analyzer import TextPipeline

# ─── Pipeline Registry ───────────────────────────────────────────────────────

PIPELINES = {
    "audio": {
        "class": AudioPipeline,
        "data_dir": "audio/data",
        "output_dir": "audio/output",
    },
    "pdf": {
        "class": PDFPipeline,
        "data_dir": "pdf/data",
        "output_dir": "pdf/output",
    },
    "image": {
        "class": ImagePipeline,
        "data_dir": "images/data",
        "output_dir": "images/output",
    },
    "video": {
        "class": VideoPipeline,
        "data_dir": "video/data",
        "output_dir": "video/output",
    },
    "text": {
        "class": TextPipeline,
        "data_dir": "text/data",
        "output_dir": "text/output",
    },
}


def run_single(module_name: str) -> pd.DataFrame:
    """Run a single module's pipeline through all 5 stages."""
    if module_name not in PIPELINES:
        print(f"Unknown module: {module_name}. Valid: {list(PIPELINES.keys())}")
        return pd.DataFrame()

    config = PIPELINES[module_name]
    root = os.path.dirname(os.path.abspath(__file__))

    pipeline = config["class"](
        data_dir=os.path.join(root, config["data_dir"]),
        output_dir=os.path.join(root, config["output_dir"]),
    )

    # ── Runs Stage 1 → Stage 2 → Stage 3 → Stage 4 automatically ──
    return pipeline.run()


def merge_all_outputs() -> pd.DataFrame:
    """Stage 4 integration: merge all 5 module outputs into one dataset."""
    root = os.path.dirname(os.path.abspath(__file__))
    all_dfs = []

    output_files = [
        "audio/output/audio_results.csv",
        "pdf/output/pdf_results.csv",
        "images/output/image_results.csv",
        "video/output/video_results.csv",
        "text/output/text_results.csv",
    ]

    for csv_path in output_files:
        full_path = os.path.join(root, csv_path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            if not df.empty:
                all_dfs.append(df)
                print(f"  ✓ Loaded {csv_path} ({len(df)} records)")
        else:
            print(f"  ✗ Missing: {csv_path}")

    if not all_dfs:
        print("No output files found. Run individual modules first.")
        return pd.DataFrame()

    # Merge on Incident_ID (outer join to keep all records)
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = merged.merge(df, on="Incident_ID", how="outer", suffixes=("", "_dup"))

    # Remove duplicate columns
    merged = merged.loc[:, ~merged.columns.str.endswith("_dup")]

    # Fill missing values
    merged = merged.fillna("")

    # Save final integrated output
    output_dir = os.path.join(root, "integration", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_incident_report.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n  ✓ Final merged dataset: {output_path} ({len(merged)} records)")
    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    modules_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(PIPELINES.keys())

    print("=" * 60)
    print("  MULTIMODAL INCIDENT REPORT ANALYZER")
    print("=" * 60)

    # Run each selected module through the common pipeline
    for module in modules_to_run:
        if module == "merge":
            continue
        print(f"\n▶ Running {module.upper()} module...")
        run_single(module)

    # If running all modules (or explicitly requested), merge outputs
    if len(modules_to_run) == len(PIPELINES) or "merge" in sys.argv:
        print("\n" + "=" * 60)
        print("  MERGING ALL OUTPUTS (Integration)")
        print("=" * 60)
        merge_all_outputs()

    print("\n✅ Pipeline complete.")
