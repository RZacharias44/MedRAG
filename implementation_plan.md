Action Plan: Adapting MedRAG to Run with the DDXPlus Dataset (v3)

This document outlines the necessary steps to modify the MedRAG codebase, shifting its focus from the unavailable private CPDD (Chronic Pain Diagnostic Dataset) to the public DDXPlus dataset. This version implements the **exact preprocessing and sampling methodology described in the original paper** to ensure a faithful replication.

## Paper's Preprocessing Methodology

From the paper:
> "DDXPlus We directly use the training set and test set in a split dataset in the ratio of 8:1:1(validation set). Due to the massive size of the dataset with over a million synthesized patients' records, which is too large for the scale of our task, we first fixed the number of samples in the test set to 30, which corresponds to the fewest pathology. For the other pathology with more samples, we randomly select 30 samples to form the whole test set. In the training set, we randomly pick 240 samples for each pathology to retrieve. This approach can ensure we get a maximum balanced sub-dataset containing 13230 patients' EHR in total. The random seed is set to 42."

**Key Parameters:**
- Training set: **240 samples per pathology**
- Test set: **30 samples per pathology**  
- Random seed: **42** (for reproducibility)
- Total expected: **13,230 patients** (49 pathologies × 270 samples per pathology)
- Data split: 8:1:1 ratio (train:validate:test files combined, then resampled)

---

## Phase 1: Data Preparation & Sampling

This phase focuses on downloading the raw DDXPlus data and then creating the specific, balanced subset of patient records that the authors used for their experiments.

### Step 1: Download the DDXPlus Dataset

- **Action:** Go to the DDXPlus Figshare page linked in the project's README.md.
- **Download:** Download the complete dataset. You should get three CSV files:
  - `release_train_patients.csv`
  - `release_validate_patients.csv`
  - `release_test_patients.csv`
- Place these files in the `./dataset/` directory.

### Step 2: Preprocess and Sample the Dataset

**This is the critical step.** The preprocessing script (`preprocess_ddxplus.py`) implements the paper's exact methodology:

**What the script does:**

1. **Loads all DDXPlus CSV files** (train, validate, test) and combines them
2. **Groups by pathology** and samples:
   - 240 samples per pathology for training
   - 30 samples per pathology for testing
   - Uses random seed 42 for reproducibility
3. **Skips pathologies** with insufficient samples (< 270 total)
4. **Writes individual JSON files** for each patient record
5. **Generates ground truth CSV** for evaluation

**How to run:**

```bash
# Ensure you have the required CSV files in ./dataset/
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv run preprocess_ddxplus.py
```

**Expected output:**
- Training JSONs: `./dataset/DDXPlus/train/participant_*.json`
- Test JSONs: `./dataset/DDXPlus/test/participant_*.json`
- Ground truth: `./dataset/DDXPlus_ground_truth.csv`
- Should produce ~13,230 total samples (49 pathologies × 270 samples)

### Step 3: Validate Data Consistency

After preprocessing, run the automated validation script to verify data quality:

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv run validate_preprocessing.py
```

**What the validation script checks:**

1. ✅ **File Counts**: Verifies correct number of train/test files (11,760 + 1,470 = 13,230)
2. ✅ **Ground Truth Structure**: Validates CSV format and required columns
3. ✅ **JSON Structure**: Checks that patient JSON files have all required fields
4. ✅ **Pathology Distribution**: Ensures balanced dataset (240 train + 30 test per pathology)
5. ✅ **Knowledge Graph Consistency**: Confirms all pathologies exist in the KG

**Expected validation output:**
```
======================================================================
DDXPlus Preprocessing Validation
======================================================================

Step 1: Validating File Counts
======================================================================
✓ Training files: 11760 (expected: 11760)
✓ Test files: 1470 (expected: 1470)
✓ Total samples: 13230 (expected: 13230)

Step 2: Validating Ground Truth File
======================================================================
✓ Ground truth file loaded: 1470 entries
✓ All required columns present
✓ No duplicate participant numbers
✓ Pathology count matches expected: 49
✓ All pathologies have exactly 30 test samples (balanced)

Step 3: Validating JSON Structure
======================================================================
✓ All 40 sampled JSON files have correct structure

Step 4: Validating Pathology Distribution
======================================================================
✓ All pathologies appear in both train and test sets
✓ All training pathologies have exactly 240 samples
✓ All test pathologies have exactly 30 samples

Step 5: Validating Against Knowledge Graph
======================================================================
✓ Knowledge graph loaded: 49 conditions
✓ All dataset pathologies exist in knowledge graph

Validation Summary
======================================================================
✓ File counts
✓ Ground truth structure
✓ JSON structure
✓ Pathology distribution
✓ Knowledge graph consistency

Overall: 5/5 checks passed

🎉 All validations passed! Data preprocessing is correct.
```

---

## Phase 2: Code Configuration ✅ COMPLETE

The preprocessing script has placed files in the correct locations. All code has been updated to work with DDXPlus dataset.

### ✅ Step 1: File Paths in `authentication.py` (Already Correct)

Current configuration:

```python
# In authentication.py
ob_path='./dataset/DDXPlus/train'
test_folder_path="./dataset/DDXPlus/test"
ground_truth_file_path='./dataset/DDXPlus_ground_truth.csv'
augmented_features_path='./dataset/knowledge graph of DDXPlus.xlsx'
```

**Status:** ✓ No changes needed

### ✅ Step 2: Update Paths and API Key in `KG_Retrieve.py` (COMPLETED)

**Changes made:**

```python
# In KG_Retrieve.py
from authentication import api_key, augmented_features_path, ground_truth_file_path

KG_file_path = augmented_features_path
file_path = ground_truth_file_path
client = openai.OpenAI(api_key=api_key)
```

**Benefits:** Centralized configuration, no hardcoded credentials

### ✅ Step 3: Update System Prompt in `main_MedRAG.py` (COMPLETED)

**Changes made:**

1. **Updated disease list** with all 49 DDXPlus pathologies (exact French names):
   ```
   Anaphylaxie, Angine instable, Angine stable, Anémie, Asthme exacerbé ou bronchospasme, 
   Attaque de panique, Bronchiectasies, Bronchiolite, Bronchite, Chagas, Coqueluche, 
   Céphalée en grappe, Ebola, Embolie pulmonaire, ...
   ```

2. **Updated output format** from pain-specific to general diagnostics:
   - Diagnosis with exact French name
   - Clinical reasoning and differential diagnosis
   - Follow-up questions
   - Clinical recommendations

**See:** `PHASE2_COMPLETION_REPORT.md` for complete details

---

## Phase 3: Execution

With the data correctly sampled and the code configured, you are ready to run the replication.

**Steps:**
1. Ensure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh` (if not already installed)
2. Install dependencies: `uv sync` (reads from `pyproject.toml` and `uv.lock`)
3. Set API keys: Add your OpenAI API key in `authentication.py`
4. Run the main script: `uv run main.py`

**Note:** By following this plan, you are precisely recreating the experimental conditions described in the paper, which is the gold standard for a replication study. The preprocessing ensures:
- ✅ Balanced dataset (240 train + 30 test per pathology)
- ✅ Reproducible sampling (random seed 42)
- ✅ Correct scale (~13,230 patients from 49 pathologies)
- ✅ Paper-compliant methodology