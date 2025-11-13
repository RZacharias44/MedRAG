# Phase 3: Execution Guide

## Overview

Phase 3 involves running the MedRAG pipeline on the DDXPlus dataset. This guide provides step-by-step instructions for setting up and executing the experiments.

## Prerequisites

✅ **Phase 1:** Data preprocessing complete (13,230 patients)
✅ **Phase 2:** Code configuration complete (DDXPlus paths and prompts)
✅ **Phase 3:** Ready to execute

## Setup Steps

### Step 1: Configure API Keys

MedRAG requires OpenAI API for embeddings and LLM inference.

#### Option A: Using .env File (Recommended)

1. **Copy the example file:**
   ```bash
   cd "/Users/sunray/Documents/masters thesis/MedRAG"
   cp .env.example .env
   ```

2. **Edit .env with your keys:**
   ```bash
   # Open in your preferred editor
   nano .env
   # or
   code .env
   ```

3. **Add your API keys:**
   ```env
   OPENAI_API_KEY=sk-your-actual-key-here
   HUGGINGFACE_TOKEN=hf_your-actual-token-here
   ```

4. **Save and close** (Ctrl+X, then Y in nano)

#### Option B: Export as Environment Variables

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
export HUGGINGFACE_TOKEN="hf_your-actual-token-here"
```

**Note:** Option A is preferred as the keys persist across terminal sessions.

#### Getting API Keys

**OpenAI:**
- Visit: https://platform.openai.com/api-keys
- Create a new API key
- **Important:** Add billing information to your OpenAI account

**Hugging Face (Optional - only for open-source models):**
- Visit: https://huggingface.co/settings/tokens
- Create a new token with read access

### Step 2: Install Dependencies

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv sync
```

This installs all required packages from `pyproject.toml` and `uv.lock`.

### Step 3: Verify Setup

Check that everything is configured correctly:

```bash
# Verify API key is loaded
uv run python3 -c "from authentication import api_key; print('API Key loaded:', api_key[:7] + '...' if api_key else 'NOT SET')"

# Verify data files exist
ls -lh dataset/DDXPlus/train/*.json | wc -l  # Should show 11760
ls -lh dataset/DDXPlus/test/*.json | wc -l   # Should show 1470

# Verify ground truth
head -5 dataset/DDXPlus_ground_truth.csv
```

Expected output:
```
API Key loaded: sk-proj...
11760
1470
Participant No.,Processed Diagnosis,Diagnoses (related to pain)
1,Péricardite,Péricardite
...
```

## Execution Options

### Option 1: Single Patient Test (Recommended First)

Test the pipeline on one patient to verify everything works:

1. **Modify main.py temporarily:**
   ```python
   # Line 24: Change sample range to test just one patient
   samplerange=range(1,2)  # Test participant_1 only
   ```

2. **Run:**
   ```bash
   uv run main.py
   ```

3. **Expected behavior:**
   - Loads patient_1.json from test set
   - Generates embeddings (if first run)
   - Retrieves similar training cases via FAISS
   - Calls OpenAI API for diagnosis
   - Saves result to CSV

4. **Expected output:**
   ```
   topk: 1
   top_ns: 1
   match_n: 5
   i= 1
   {'Participant No.': 1, 'Processed Diagnosis': 'Péricardite', ...}
   load existing embeddings... (or) generate new embeddings...
   index:  [[5432]]
   Additional Info: ...
   Success!!!
   ________________________________________________________________
   ```

5. **Check results:**
   ```bash
   cat test_results_topk1_topn1_matchn5_range\(1,\ 2\)_MedRAG.csv
   ```

### Option 2: Small Batch Test

Test on 10 patients:

```python
# In main.py
samplerange=range(1,11)  # Test first 10 patients
```

### Option 3: Full Test Set (1,470 patients)

Run on the complete test set:

```python
# In main.py
samplerange=range(1,1471)  # All 1,470 test patients
```

**Warning:** This will take several hours and cost ~$50-100 in API fees.

### Option 4: Specific Pathology Subset

Test on patients with a specific diagnosis:

```python
import pandas as pd

# Load ground truth
gt = pd.read_csv('dataset/DDXPlus_ground_truth.csv')

# Get patients with a specific diagnosis
target_diagnosis = 'Pneumothorax spontané'
target_patients = gt[gt['Processed Diagnosis'] == target_diagnosis]['Participant No.'].tolist()

# In main.py
samplerange = target_patients  # e.g., [1, 45, 123, ...]
```

## Runtime and Cost Estimates

### First Run (With Embedding Generation)

| Component | Count | Time | Cost |
|-----------|-------|------|------|
| Training embeddings | 11,760 | ~30 min | $15-20 |
| KG symptom embeddings | ~500 | ~2 min | $0.10 |
| Test patient (1) | 1 | ~10 sec | $0.01 |
| **First run total** | | **~35 min** | **~$20** |

### Subsequent Runs (Embeddings Cached)

| Component | Patients | Time | Cost |
|-----------|----------|------|------|
| Single patient | 1 | ~10 sec | $0.01-0.02 |
| 10 patients | 10 | ~2 min | $0.10-0.20 |
| 100 patients | 100 | ~20 min | $1-2 |
| Full test set | 1,470 | ~4-5 hours | $15-30 |

**Cost breakdown per patient:**
- Embedding query: $0.0001
- GPT-4o generation: $0.01-0.02
- Total per patient: ~$0.01-0.02

### Model Selection Impact

| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| gpt-4o | Fast | $$$ | Best |
| gpt-4o-mini | Faster | $$ | Good |
| gpt-3.5-turbo | Fastest | $ | Moderate |
| Llama 3.1 (HF) | Moderate | Free* | Good |

*Free tier available, paid for faster inference

Change model in `main.py`:
```python
# In generate_diagnosis_report() call
model='gpt-4o'  # or 'gpt-4o-mini', 'gpt-3.5-turbo-0125'
```

## Monitoring Progress

### During Execution

The script prints progress information:
```
topk: 1
top_ns: 1
match_n: 5
i= 123
Processing patient_123...
index: [[5432]]
Additional Info: Pneumonie has symptom dyspnée, toux...
Success!!!
________________________________________________________________
```

### Check Generated Results

Results are saved to CSV:
```bash
# View results
cat test_results_topk1_topn1_matchn5_range\(1,\ 1471\)_MedRAG.csv

# Count processed patients
wc -l test_results_topk1_topn1_matchn5_range\(1,\ 1471\)_MedRAG.csv
```

Result CSV format:
```csv
Participant No.,Generated Diagnosis,True Diagnosis,Ori Truth,Generated report
1,Péricardite,Péricardite,Péricardite,"### Diagnoses\n1. **Diagnosis**: Péricardite..."
```

## Troubleshooting

### Issue: "No module named 'authentication'"

**Solution:**
```bash
# Make sure you're running from the project root
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv run main.py
```

### Issue: "OpenAI API key not found"

**Solution:**
1. Check .env file exists: `ls -la .env`
2. Verify key format: `cat .env`
3. Try export: `export OPENAI_API_KEY="your-key"`

### Issue: "RateLimitError: You exceeded your current quota"

**Solution:**
- Add billing information to OpenAI account
- Check usage at: https://platform.openai.com/usage
- Reduce batch size or add delays

### Issue: "FileNotFoundError: participant_X.json"

**Solution:**
- Some participant numbers might be missing
- The code already handles this: `if not os.path.exists(file_path): continue`
- Check which files exist: `ls dataset/DDXPlus/test/*.json | head`

### Issue: "Memory Error" or "OOM"

**Solution:**
- Reduce sample range
- Process in smaller batches
- Close other applications
- Increase system swap space

### Issue: Slow FAISS retrieval

**Solution:**
- First run is slow (generating embeddings)
- Check if embeddings are cached: `ls -lh Embeddings_saved/`
- Subsequent runs use cached embeddings (much faster)

## Results Analysis

After execution, analyze results using the metrics script:

```bash
uv run python metrics/metrics\ DDXPlus.py
```

Expected metrics:
- **Accuracy @ L3** (disease-level): Target >85%
- **Accuracy @ L2** (subcategory): Target >90%
- **Accuracy @ L1** (category): Target >95%

## Best Practices

### 1. Start Small
- Test with 1 patient first
- Then 10 patients
- Finally full test set

### 2. Monitor Costs
- Check OpenAI usage dashboard regularly
- Set billing alerts
- Start with gpt-4o-mini for testing

### 3. Save Intermediate Results
- Results are saved incrementally to CSV
- If interrupted, you can resume from last patient
- Keep backups of result files

### 4. Version Control
- Commit result CSVs with meaningful names
- Document any parameter changes
- Track which model/config produced which results

## Parameter Tuning

### Key Parameters in main.py

```python
topk=1      # Number of similar training cases to retrieve
top_n=1     # Top N symptoms to match in KG
match_n=5   # Number of similar symptoms to find
model='gpt-4o'  # LLM model for diagnosis generation
```

**Experiment suggestions:**
- Try `topk=3` to retrieve more training cases
- Try `top_n=3` for more KG context
- Try `match_n=10` for broader symptom matching

## Success Criteria

Phase 3 is complete when:
- ✅ Pipeline runs without errors
- ✅ Diagnoses are generated in correct French format
- ✅ Results CSV contains all test patients
- ✅ Accuracy metrics are calculated
- ✅ Results are comparable to paper (Target: >85% @ L3)

## Next Steps After Phase 3

1. **Analyze Results**
   - Calculate accuracy metrics
   - Compare with paper results
   - Identify error patterns

2. **Error Analysis**
   - Review misdiagnosed cases
   - Check if KG information helped
   - Analyze confidence scores

3. **Optimization**
   - Tune retrieval parameters
   - Experiment with different models
   - Improve prompt engineering

4. **Documentation**
   - Write up findings for thesis
   - Create visualizations
   - Document limitations

## Quick Reference Commands

```bash
# Full workflow from scratch
cd "/Users/sunray/Documents/masters thesis/MedRAG"
cp .env.example .env
# Edit .env with your keys
uv sync
uv run validate_preprocessing.py  # Verify data
uv run main.py  # Run pipeline

# Check results
ls -lh test_results_*.csv
head test_results_*.csv
```

---

**You're ready to run MedRAG on DDXPlus!** 🚀

Start with a single patient test, then scale up once confirmed working.

