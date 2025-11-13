# Dataset Directory

## Overview

This directory contains the DDXPlus dataset files used for the MedRAG experiments. Due to size constraints, **large data files are not tracked in git** and must be regenerated using the preprocessing script.

## File Status

### ✅ Tracked in Git (Small Reference Files)

These files ARE committed to the repository:

- `release_conditions.json` - DDXPlus conditions/pathologies metadata
- `release_evidences.json` - DDXPlus symptoms/evidence metadata  
- `knowledge graph of DDXPlus.xlsx` - Knowledge graph structure

### ❌ Not Tracked in Git (Large Data Files)

These files are **ignored by git** and must be generated/downloaded:

- `release_train_patients.csv` (~500MB) - Raw training data
- `release_test_patients.csv` (~100MB) - Raw test data
- `release_validate_patients.csv` (~100MB) - Raw validation data
- `DDXPlus_ground_truth.csv` - Preprocessed ground truth labels
- `DDXPlus/train/*.json` - 11,760 preprocessed training files
- `DDXPlus/test/*.json` - 1,470 preprocessed test files

## How to Obtain the Data

### Step 1: Download Raw DDXPlus Data

Download the three large CSV files from the DDXPlus dataset source and place them in this directory:

- `release_train_patients.csv`
- `release_test_patients.csv`
- `release_validate_patients.csv`

**Source:** Check the main README.md for the DDXPlus dataset link.

### Step 2: Run Preprocessing

From the project root, run:

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv run preprocess_ddxplus.py
```

This will generate:
- `DDXPlus_ground_truth.csv`
- `DDXPlus/train/` directory with 11,760 JSON files
- `DDXPlus/test/` directory with 1,470 JSON files

### Step 3: Validate

Verify the preprocessing was successful:

```bash
uv run validate_preprocessing.py
```

Expected result: `5/5 checks passed`

## Why Are These Files Ignored?

Git repositories have size limits, and these data files total over **700MB**. The large files are:

1. **Too big for GitHub** (100MB file size limit per file)
2. **Reproducible** - Can be regenerated from the raw CSVs using the preprocessing script
3. **Deterministic** - Random seed 42 ensures identical results

## Directory Structure

```
dataset/
├── README.md                              # This file
├── release_conditions.json                # ✅ Tracked
├── release_evidences.json                 # ✅ Tracked
├── knowledge graph of DDXPlus.xlsx        # ✅ Tracked
├── release_train_patients.csv             # ❌ Not tracked (download)
├── release_test_patients.csv              # ❌ Not tracked (download)
├── release_validate_patients.csv          # ❌ Not tracked (download)
├── DDXPlus_ground_truth.csv               # ❌ Not tracked (generated)
└── DDXPlus/                               # ❌ Not tracked (generated)
    ├── train/                             # 11,760 JSON files
    │   ├── participant_1.json
    │   ├── participant_2.json
    │   └── ...
    └── test/                              # 1,470 JSON files
        ├── participant_1.json
        ├── participant_2.json
        └── ...
```

## For Collaborators

When cloning this repository, you'll need to:

1. Download the raw DDXPlus CSV files (see Step 1 above)
2. Run the preprocessing script (see Step 2 above)
3. Validate the results (see Step 3 above)

This ensures everyone is working with the exact same preprocessed data (thanks to the fixed random seed).

