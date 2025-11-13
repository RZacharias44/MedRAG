"""
DDXPlus Preprocessing Script
============================
This script implements the exact preprocessing methodology described in the paper:

From the paper:
"Due to the massive size of the dataset with over a million synthesized patients' 
records, which is too large for the scale of our task, we first fixed the number 
of samples in the test set to 30, which corresponds to the fewest pathology. For 
the other pathology with more samples, we randomly select 30 samples to form the 
whole test set. In the training set, we randomly pick 240 samples for each 
pathology to retrieve. This approach can ensure we get a maximum balanced 
sub-dataset containing 13230 patients' EHR in total. The random seed is set to 42."

Expected output: 
- Training set: 240 samples per pathology
- Test set: 30 samples per pathology  
- Total: 13,230 patients (49 pathologies × 270 samples)
- Random seed: 42

Usage:
    uv run preprocess_ddxplus.py
"""

import os
import json
import pandas as pd
from typing import Tuple

# CONFIGURATION
TRAIN_SAMPLES_PER_PATHOLOGY = 240
TEST_SAMPLES_PER_PATHOLOGY = 30
RANDOM_SEED = 42

# Input files (already unpacked CSVs from DDXPlus dataset)
TRAIN_CSV = './dataset/release_train_patients.csv'
TEST_CSV = './dataset/release_test_patients.csv'
VALIDATE_CSV = './dataset/release_validate_patients.csv'

# Output locations
TRAIN_DIR = './dataset/DDXPlus/train'
TEST_DIR = './dataset/DDXPlus/test'
GROUND_TRUTH_FILE = './dataset/DDXPlus_ground_truth.csv'

# Ensure output dirs
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Column names in DDXPlus CSVs
COL_AGE = 'AGE'
COL_DIFF = 'DIFFERENTIAL_DIAGNOSIS'
COL_SEX = 'SEX'
COL_PATHOLOGY = 'PATHOLOGY'
COL_EVIDENCES = 'EVIDENCES'
COL_INITIAL = 'INITIAL_EVIDENCE'


def load_and_combine_data() -> pd.DataFrame:
    """
    Load all DDXPlus CSV files and combine them.
    The paper uses an 8:1:1 split, so we combine all data first,
    then sample from the combined pool.
    """
    print("Loading DDXPlus data files...")
    dfs = []
    
    if os.path.exists(TRAIN_CSV):
        print(f"  Loading {TRAIN_CSV}...")
        dfs.append(pd.read_csv(TRAIN_CSV))
    
    if os.path.exists(VALIDATE_CSV):
        print(f"  Loading {VALIDATE_CSV}...")
        dfs.append(pd.read_csv(VALIDATE_CSV))
        
    if os.path.exists(TEST_CSV):
        print(f"  Loading {TEST_CSV}...")
        dfs.append(pd.read_csv(TEST_CSV))
    
    if not dfs:
        raise FileNotFoundError("No DDXPlus CSV files found. Please check file paths.")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total records")
    return combined_df


def sample_balanced_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample a balanced dataset following the paper's methodology:
    - 30 samples per pathology for test set
    - 240 samples per pathology for training set
    - Random seed: 42
    
    Returns: (train_df, test_df)
    """
    print(f"\nSampling balanced dataset (seed={RANDOM_SEED})...")
    print(f"  Target: {TRAIN_SAMPLES_PER_PATHOLOGY} train + {TEST_SAMPLES_PER_PATHOLOGY} test per pathology")
    
    grouped = df.groupby(COL_PATHOLOGY)
    total_pathologies = len(grouped)
    print(f"  Found {total_pathologies} unique pathologies")
    
    train_samples = []
    test_samples = []
    skipped_pathologies = []
    
    for pathology_name, group in grouped:
        min_required = TRAIN_SAMPLES_PER_PATHOLOGY + TEST_SAMPLES_PER_PATHOLOGY
        
        if len(group) < min_required:
            skipped_pathologies.append((pathology_name, len(group)))
            continue
        
        # Sample with fixed random seed for reproducibility
        sampled = group.sample(n=min_required, random_state=RANDOM_SEED)
        
        # Split into test and train
        # Use the same random seed to ensure reproducibility
        test_subset = sampled.sample(n=TEST_SAMPLES_PER_PATHOLOGY, random_state=RANDOM_SEED)
        train_subset = sampled.drop(test_subset.index)
        
        test_samples.append(test_subset)
        train_samples.append(train_subset)
    
    if skipped_pathologies:
        print(f"\n  Warning: Skipped {len(skipped_pathologies)} pathologies with insufficient samples:")
        for name, count in skipped_pathologies[:5]:  # Show first 5
            print(f"    - {name}: {count} samples (need {min_required})")
        if len(skipped_pathologies) > 5:
            print(f"    ... and {len(skipped_pathologies) - 5} more")
    
    train_df = pd.concat(train_samples, ignore_index=True)
    test_df = pd.concat(test_samples, ignore_index=True)
    
    # Shuffle to avoid any ordering bias
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    total_samples = len(train_df) + len(test_df)
    num_pathologies_used = len(train_samples)
    
    print(f"\n  Results:")
    print(f"    Training samples: {len(train_df)}")
    print(f"    Test samples: {len(test_df)}")
    print(f"    Total samples: {total_samples}")
    print(f"    Pathologies used: {num_pathologies_used}")
    print(f"    Expected total (paper): 13,230 (49 pathologies)")
    
    return train_df, test_df


def to_patient_json(row: pd.Series, participant_no: int) -> dict:
    """Convert a DataFrame row to a patient JSON structure."""
    return {
        'Participant No.': participant_no,
        'Processed Diagnosis': str(row[COL_PATHOLOGY]),
        'Diagnoses (related to pain)': str(row[COL_PATHOLOGY]),
        'Age': int(row[COL_AGE]) if pd.notna(row[COL_AGE]) else None,
        'Sex': str(row[COL_SEX]) if pd.notna(row[COL_SEX]) else None,
        'Differential Diagnosis': str(row[COL_DIFF]) if pd.notna(row[COL_DIFF]) else '',
        'Evidences': str(row[COL_EVIDENCES]) if pd.notna(row[COL_EVIDENCES]) else '',
        'Initial Evidence': str(row[COL_INITIAL]) if pd.notna(row[COL_INITIAL]) else '',
        # Compatibility fields for downstream code
        'Pain Presentation and Description Areas of pain as per physiotherapy input': '',
        'Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles': ''
    }


def write_split(df: pd.DataFrame, out_dir: str, start_id: int = 1) -> int:
    """Write DataFrame rows as individual JSON files."""
    print(f"\n  Writing {len(df)} JSON files to {out_dir}...")
    
    next_id = start_id
    for idx, row in df.iterrows():
        patient = to_patient_json(row, next_id)
        out_path = os.path.join(out_dir, f'participant_{next_id}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(patient, f, indent=2)
        next_id += 1
    
    print(f"  ✓ Wrote {len(df)} files")
    return next_id


def build_ground_truth(test_df: pd.DataFrame) -> pd.DataFrame:
    """Build ground truth CSV from test DataFrame."""
    print("\n  Building ground truth CSV...")
    
    records = []
    participant_no = 1
    for _, row in test_df.iterrows():
        records.append({
            'Participant No.': participant_no,
            'Processed Diagnosis': str(row[COL_PATHOLOGY]),
            'Diagnoses (related to pain)': str(row[COL_PATHOLOGY])
        })
        participant_no += 1
    
    gt_df = pd.DataFrame(records)
    print(f"  ✓ Created ground truth with {len(gt_df)} entries")
    return gt_df


def main():
    print("="*70)
    print("DDXPlus Dataset Preprocessing (Paper Methodology)")
    print("="*70)
    
    # Step 1: Load and combine all data
    combined_df = load_and_combine_data()
    
    # Step 2: Sample balanced dataset (240 train + 30 test per pathology, seed=42)
    train_df, test_df = sample_balanced_dataset(combined_df)
    
    # Step 3: Write training set JSONs
    print(f"\nWriting training set...")
    write_split(train_df, TRAIN_DIR, start_id=1)
    
    # Step 4: Write test set JSONs  
    print(f"\nWriting test set...")
    write_split(test_df, TEST_DIR, start_id=1)
    
    # Step 5: Generate ground truth CSV
    print(f"\nGenerating ground truth...")
    gt = build_ground_truth(test_df)
    gt.to_csv(GROUND_TRUTH_FILE, index=False)
    print(f"  ✓ Saved to {GROUND_TRUTH_FILE}")
    
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"Training JSONs: {TRAIN_DIR}")
    print(f"Test JSONs: {TEST_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_FILE}")
    print("="*70)


if __name__ == '__main__':
    main()
