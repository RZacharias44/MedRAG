import os
import json
import pandas as pd
from typing import Tuple

# Input files (already unpacked CSVs)
TRAIN_CSV = './dataset/DDXplus/release_train_patients'
TEST_CSV = './dataset/DDXplus/release_test_patients'
VALIDATE_CSV = './dataset/DDXplus/release_validate_patients'

# Output locations
TRAIN_DIR = './dataset/DDXPlus/train'
TEST_DIR = './dataset/DDXPlus/test'
GROUND_TRUTH_FILE = './dataset/DDXPlus_ground_truth.csv'

# Ensure output dirs
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Column names in DDXPlus CSVs
# AGE, DIFFERENTIAL_DIAGNOSIS, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE
COL_AGE = 'AGE'
COL_DIFF = 'DIFFERENTIAL_DIAGNOSIS'
COL_SEX = 'SEX'
COL_PATHOLOGY = 'PATHOLOGY'
COL_EVIDENCES = 'EVIDENCES'
COL_INITIAL = 'INITIAL_EVIDENCE'


def load_ddxplus_csv(path: str) -> pd.DataFrame:
    # Files are large; use pandas defaults (they're CSV-like without extension)
    return pd.read_csv(path)


def to_patient_json(row: pd.Series, participant_no: int) -> dict:
    # Map to keys compatible with downstream code
    # Keep both raw and minimally transformed fields for flexibility
    return {
        'Participant No.': participant_no,
        'Processed Diagnosis': str(row[COL_PATHOLOGY]),
        'Diagnoses (related to pain)': str(row[COL_PATHOLOGY]),
        'Age': int(row[COL_AGE]) if pd.notna(row[COL_AGE]) else None,
        'Sex': str(row[COL_SEX]) if pd.notna(row[COL_SEX]) else None,
        'Differential Diagnosis': str(row[COL_DIFF]) if pd.notna(row[COL_DIFF]) else '',
        'Evidences': str(row[COL_EVIDENCES]) if pd.notna(row[COL_EVIDENCES]) else '',
        'Initial Evidence': str(row[COL_INITIAL]) if pd.notna(row[COL_INITIAL]) else '',
        # Minimal compatibility fields expected elsewhere:
        'Pain Presentation and Description Areas of pain as per physiotherapy input': '',
        'Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles': ''
    }


def write_split(df: pd.DataFrame, out_dir: str, start_id: int) -> int:
    next_id = start_id
    for _, row in df.iterrows():
        patient = to_patient_json(row, next_id)
        out_path = os.path.join(out_dir, f'participant_{next_id}.json')
        with open(out_path, 'w') as f:
            json.dump(patient, f)
        next_id += 1
    return next_id


def build_ground_truth(test_df: pd.DataFrame) -> pd.DataFrame:
    # Ground-truth columns expected by KG_Retrieve/main
    # Use sequential Participant No. aligned with test JSONs
    records = []
    participant_no = 1
    for _, row in test_df.iterrows():
        records.append({
            'Participant No.': participant_no,
            'Processed Diagnosis': str(row[COL_PATHOLOGY]),
            'Diagnoses (related to pain)': str(row[COL_PATHOLOGY])
        })
        participant_no += 1
    return pd.DataFrame(records)


def main():
    train_df = load_ddxplus_csv(TRAIN_CSV)
    test_df = load_ddxplus_csv(TEST_CSV)

    # Write JSONs. Ensure Participant No. sequences match ground truth for test.
    # Train: start from 1 but independent of test indexing
    _ = write_split(train_df, TRAIN_DIR, start_id=1)

    # Test: start from 1 to match ground truth construction
    _ = write_split(test_df, TEST_DIR, start_id=1)

    gt = build_ground_truth(test_df)
    gt.to_csv(GROUND_TRUTH_FILE, index=False)

    print(f'Wrote {len(train_df)} train JSONs to {TRAIN_DIR}')
    print(f'Wrote {len(test_df)} test JSONs to {TEST_DIR}')
    print(f'Ground truth saved to {GROUND_TRUTH_FILE}')


if __name__ == '__main__':
    main()
