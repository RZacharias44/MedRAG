# Quick Start Guide - DDXPlus Preprocessing & Validation

## Overview

This guide shows you how to preprocess the DDXPlus dataset according to the paper's methodology and validate the results.

## Prerequisites

1. **uv package manager** (recommended for this project)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **DDXPlus CSV files** in `./dataset/`:
   - `release_train_patients.csv`
   - `release_validate_patients.csv`
   - `release_test_patients.csv`

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv sync
```

This will install all required packages from `pyproject.toml` and `uv.lock`.

### 2. Run Preprocessing

```bash
uv run preprocess_ddxplus.py
```

**What happens:**
- Loads and combines all DDXPlus CSV files (~1M records)
- Samples 240 train + 30 test per pathology (random seed 42)
- Creates ~13,230 individual patient JSON files
- Generates ground truth CSV for evaluation
- **Runtime:** ~1-2 minutes

**Output:**
- `./dataset/DDXPlus/train/` - 11,760 training JSON files
- `./dataset/DDXPlus/test/` - 1,470 test JSON files  
- `./dataset/DDXPlus_ground_truth.csv` - Ground truth labels

### 3. Validate Preprocessing

```bash
uv run validate_preprocessing.py
```

**What it checks:**
1. ✅ File counts (11,760 train + 1,470 test = 13,230 total)
2. ✅ Ground truth structure and format
3. ✅ JSON file structure (all required fields present)
4. ✅ Balanced distribution (240 train + 30 test per pathology)
5. ✅ Knowledge graph consistency (all pathologies exist in KG)

**Expected result:**
```
Overall: 5/5 checks passed
🎉 All validations passed! Data preprocessing is correct.
```

## Paper Methodology Confirmation

✅ **Training set:** 240 samples per pathology  
✅ **Test set:** 30 samples per pathology  
✅ **Random seed:** 42 (for reproducibility)  
✅ **Total samples:** 13,230 (49 pathologies × 270 samples)  
✅ **Data combination:** 8:1:1 split files combined, then resampled  

## Troubleshooting

### Issue: "No DDXPlus CSV files found"
**Solution:** Ensure the three CSV files are in `./dataset/` directory

### Issue: "Pathology count: X (expected: 49)"
**Possible causes:**
- Some pathologies don't have enough samples (need 270 minimum)
- Check the preprocessing output for skipped pathologies

### Issue: "Unbalanced pathologies found"
**Solution:** Re-run preprocessing to ensure proper sampling with seed 42

### Issue: "Pathologies not in knowledge graph"
**Solution:** Check that `release_conditions.json` is present and contains all pathologies

## Files Generated

```
./dataset/
├── DDXPlus/
│   ├── train/
│   │   ├── participant_1.json
│   │   ├── participant_2.json
│   │   └── ... (11,760 files)
│   └── test/
│       ├── participant_1.json
│       ├── participant_2.json
│       └── ... (1,470 files)
└── DDXPlus_ground_truth.csv
```

## Next Steps

After successful preprocessing and validation:

1. Update paths in `authentication.py`:
   ```python
   ob_path='./dataset/DDXPlus/train'
   test_folder_path="./dataset/DDXPlus/test"
   ground_truth_file_path='./dataset/DDXPlus_ground_truth.csv'
   ```

2. Update paths in `KG_Retrieve.py`

3. Run the main MedRAG pipeline:
   ```bash
   uv run main.py
   ```

## Additional Documentation

- **Full implementation plan:** `implementation_plan.md`
- **Detailed verification report:** `PREPROCESSING_VERIFICATION.md`
- **Preprocessing script:** `preprocess_ddxplus.py`
- **Validation script:** `validate_preprocessing.py`

## Summary

```bash
# Complete workflow in 3 commands:
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv sync                              # Install dependencies
uv run preprocess_ddxplus.py         # Preprocess data
uv run validate_preprocessing.py     # Validate results
```

✅ All preprocessing matches the paper's exact methodology!

