# Visual Guide: How Embeddings Work in MedRAG

## Quick Answer to Your Questions

### 1. Testing small subset - still need all training embeddings?

**YES** ✅ - Even testing 1 patient requires all 11,760 training embeddings.

### 2. What are embeddings for?

**Semantic similarity search** - Finding similar patients by meaning, not keywords.

### 3. No real EHR data - will it work?

**YES** ✅ - DDXPlus IS synthetic EHR data. Pipeline works perfectly.

---

## Visual Explanation

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONE-TIME SETUP (First Run)                       │
│                     Cost: $15-20 (cached forever)                   │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Load ALL Training Documents
────────────────────────────────────
┌─────────────────────────────────┐
│ ./dataset/DDXPlus/train/        │
│  ├─ participant_1.json          │
│  ├─ participant_2.json          │
│  ├─ participant_3.json          │
│  │   ...                         │
│  └─ participant_11760.json      │
└─────────────────────────────────┘
         ↓
    11,760 JSON files loaded


Step 2: Convert to Embeddings (OpenAI text-embedding-3-large)
──────────────────────────────────────────────────────────────
Each JSON file:                       Embedding (3,072 numbers):
┌──────────────────────┐             ┌──────────────────────────┐
│ {                    │             │ [0.234, -0.567, 0.891,   │
│   "Age": 43,         │    ────►    │  0.123, -0.456, 0.789,   │
│   "Sex": "F",        │             │  ... 3,072 dimensions]   │
│   "Evidences": "..." │             │                          │
│ }                    │             │ Semantic meaning vector  │
└──────────────────────┘             └──────────────────────────┘

                ↓ (Repeat 11,760 times)

┌──────────────────────────────────────────────────────────┐
│         TRAINING EMBEDDINGS MATRIX                       │
│         Shape: (11,760 patients × 3,072 dimensions)      │
│                                                          │
│  Patient 1:    [0.234, -0.567, 0.891, ...]             │
│  Patient 2:    [0.123, -0.456, 0.789, ...]             │
│  Patient 3:    [0.345, -0.678, 0.901, ...]             │
│  ...                                                     │
│  Patient 11760: [0.456, -0.789, 0.012, ...]            │
│                                                          │
│  Saved to: ./Embeddings_saved/DDXPlus_document_embeddings.npy  │
│  Size: ~400MB                                           │
│  Cost: $15-20 (ONE TIME)                                │
└──────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│              TESTING (Each Test Patient)                            │
│              Cost: $0.02 per patient (uses cached embeddings)       │
└─────────────────────────────────────────────────────────────────────┘

Step 3: Load Test Patient
──────────────────────────
┌──────────────────────────────┐
│  participant_1.json (test)   │
│  {                           │
│    "Age": 45,                │
│    "Sex": "M",               │
│    "Evidences": "dyspnée..." │
│    "True Diagnosis": "???"   │
│  }                           │
└──────────────────────────────┘
         ↓
   Convert to embedding
         ↓
┌──────────────────────────────┐
│  Test Query Embedding        │
│  [0.567, -0.234, 0.678, ...] │
│  Cost: $0.0001               │
└──────────────────────────────┘


Step 4: FAISS Similarity Search
────────────────────────────────
                    Search across ALL 11,760 embeddings
                                    ↓
┌────────────────────────────────────────────────────────────┐
│              FAISS Index (Inner Product Search)            │
│                                                            │
│  Query: [0.567, -0.234, 0.678, ...]                      │
│                                                            │
│  Compare with ALL training embeddings:                     │
│  ┌───────────────────────────────────────────┐           │
│  │ Patient 1:    similarity = 0.65           │           │
│  │ Patient 2:    similarity = 0.43           │           │
│  │ Patient 3:    similarity = 0.91  ← Best!  │           │
│  │ ...                                        │           │
│  │ Patient 11760: similarity = 0.52          │           │
│  └───────────────────────────────────────────┘           │
│                                                            │
│  Result: Patient 3 is most similar (topk=1)               │
│  Time: Milliseconds (very fast!)                          │
└────────────────────────────────────────────────────────────┘


Step 5: Retrieve Similar Case
──────────────────────────────
┌─────────────────────────────────────┐
│  Retrieved: participant_3.json      │
│  {                                  │
│    "Processed Diagnosis":           │
│      "Pneumothorax spontané",       │
│    "Evidences": "dyspnée, ..."      │
│  }                                  │
│                                     │
│  Provides context for LLM           │
└─────────────────────────────────────┘


Step 6: LLM + Knowledge Graph → Diagnosis
──────────────────────────────────────────
┌────────────────────────────────────────────────────┐
│  Prompt to GPT-4o:                                 │
│                                                    │
│  "New patient: [Age: 45, dyspnée, ...]            │
│                                                    │
│  Similar case found: Pneumothorax spontané        │
│  [symptoms: dyspnée, chest pain...]               │
│                                                    │
│  Knowledge Graph: Pneumothorax spontané           │
│  has_symptom: dyspnée, douleur thoracique         │
│  differs_from: Pneumonie (has fever)              │
│                                                    │
│  Diagnose this patient:"                          │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│  LLM Output:                                       │
│  **Diagnosis**: Pneumothorax spontané              │
│  **Reasoning**: Patient presents with sudden       │
│  onset dyspnée and chest pain, consistent with...  │
│  Cost: $0.01-0.02                                  │
└────────────────────────────────────────────────────┘
```

---

## Why ALL Training Embeddings Are Needed

### Analogy: Restaurant Recommendation System

```
Training Set = Database of 11,760 restaurants
Each restaurant = 1 patient case

You (test patient) say: "I want Italian food, romantic atmosphere, $$$"
                        ↓
                  Convert to embedding
                        ↓
         Search ALL 11,760 restaurants in database
                        ↓
              Find most similar: "Bella Roma"
                        ↓
         Recommend based on what worked for others

You can't search only 10 restaurants when there are 11,760 in the database!
```

### What Happens If You Only Use 100 Training Embeddings?

```
Full Training Set (11,760):
├─ More diverse cases
├─ Better matches for rare conditions
├─ Higher retrieval accuracy
└─ **Better diagnosis quality**

Subset (100 only):
├─ Limited diversity
├─ May miss good matches
├─ Lower retrieval quality
└─ **Worse diagnosis accuracy**

The paper's methodology requires the full balanced dataset (240×49 = 11,760)
```

---

## The Three Types of Embeddings

### 1. Training Document Embeddings (Large, Cached)

```
┌─────────────────────────────────────────────────────┐
│  Type: Training Document Embeddings                 │
│  Count: 11,760                                      │
│  Source: ./dataset/DDXPlus/train/*.json             │
│  Generated: Once on first run                       │
│  Cached at: ./Embeddings_saved/DDXPlus_document_embeddings.npy  │
│  Size: ~400MB                                       │
│  Cost: $15-20 (one-time)                            │
│  Purpose: Enable FAISS similarity search            │
│                                                     │
│  When regenerated:                                  │
│  - First run ever                                   │
│  - If cache file deleted                            │
│  - If training data changes                         │
└─────────────────────────────────────────────────────┘
```

### 2. Test Query Embeddings (Small, Per-Patient)

```
┌─────────────────────────────────────────────────────┐
│  Type: Test Query Embeddings                        │
│  Count: 1 per test patient                          │
│  Source: Current test patient JSON                  │
│  Generated: Fresh for each test patient             │
│  Cached: No (generated on-demand)                   │
│  Size: 3,072 numbers per query                      │
│  Cost: $0.0001 per query                            │
│  Purpose: Search against training embeddings        │
│                                                     │
│  Examples:                                          │
│  - Test 1 patient: 1 query embedding                │
│  - Test 10 patients: 10 query embeddings            │
│  - Test 1,470 patients: 1,470 query embeddings      │
└─────────────────────────────────────────────────────┘
```

### 3. Knowledge Graph Embeddings (Small, Cached)

```
┌─────────────────────────────────────────────────────┐
│  Type: KG Symptom Node Embeddings                   │
│  Count: ~500 symptom nodes                          │
│  Source: Knowledge graph symptoms                   │
│  Generated: Once on first run                       │
│  Cached at: ./Embeddings_saved/DDXPlus_KG_embeddings/    │
│  Size: ~5MB                                         │
│  Cost: $0.10 (one-time)                             │
│  Purpose: Match patient symptoms to KG              │
└─────────────────────────────────────────────────────┘
```

---

## Cost Breakdown by Scenario

### Scenario 1: First Run, Test 1 Patient

```
Operation                          Cost        Time
─────────────────────────────────────────────────────
Generate 11,760 train embeddings   $15-20      25min
Generate ~500 KG embeddings        $0.10       2min
Generate 1 test query              $0.0001     <1sec
LLM diagnosis (GPT-4o)             $0.01       5sec
─────────────────────────────────────────────────────
TOTAL                              $20.11      ~30min

After this, embeddings are cached forever!
```

### Scenario 2: Second Run, Test 10 Patients

```
Operation                          Cost        Time
─────────────────────────────────────────────────────
Load cached train embeddings       $0          1sec
Load cached KG embeddings          $0          <1sec
Generate 10 test queries           $0.001      <1sec
10 LLM diagnoses (GPT-4o)          $0.10       50sec
─────────────────────────────────────────────────────
TOTAL                              $0.101      ~1min

99% cheaper than first run!
```

### Scenario 3: Full Test Set (1,470 Patients, Cached)

```
Operation                          Cost        Time
─────────────────────────────────────────────────────
Load cached train embeddings       $0          1sec
Load cached KG embeddings          $0          <1sec
Generate 1,470 test queries        $0.15       10sec
1,470 LLM diagnoses (GPT-4o)       $15-30      4hours
─────────────────────────────────────────────────────
TOTAL                              $30.15      ~4-5hr

Only paying for LLM diagnoses!
```

---

## About DDXPlus as "EHR" Data

### Real EHR vs DDXPlus (Synthetic EHR)

```
┌─────────────────────────────────────────────────────────────────┐
│                  REAL EHR (Hospital Records)                    │
├─────────────────────────────────────────────────────────────────┤
│ ✗ Privacy concerns (protected health information)               │
│ ✗ Hard to access (IRB approval needed)                          │
│ ✗ Messy data (free text, inconsistent format)                   │
│ ✗ Uncertain ground truth (diagnoses may be wrong)               │
│ ✗ Expensive to collect                                          │
│ ✗ Can't share publicly                                          │
└─────────────────────────────────────────────────────────────────┘
                            vs
┌─────────────────────────────────────────────────────────────────┐
│              DDXPlus (Synthetic EHR for Research)               │
├─────────────────────────────────────────────────────────────────┤
│ ✓ No privacy concerns (completely synthetic)                    │
│ ✓ Publicly available (anyone can download)                      │
│ ✓ Clean, structured data (JSON format)                          │
│ ✓ Known ground truth (diagnoses are certain)                    │
│ ✓ Free to use                                                   │
│ ✓ Can share results publicly                                    │
│ ✓ 1M+ synthetic patient records                                 │
│ ✓ Based on real medical knowledge                               │
└─────────────────────────────────────────────────────────────────┘
```

### DDXPlus Data Structure (What Gets Embedded)

```json
{
  "Participant No.": 1,
  "Age": 43,
  "Sex": "F",
  "Evidences": [
    "douleurxx",                    ← Symptom: chest pain
    "douleurxx_carac_@_vive",       ← Characteristic: sharp
    "douleurxx_endroitducorps_@_haut_du_thorax",  ← Location
    "dyspn",                        ← Symptom: dyspnea
    "palpit"                        ← Symptom: palpitations
  ],
  "Initial Evidence": "palpit",      ← Chief complaint
  "Differential Diagnosis": [         ← DDX list with probabilities
    ["Péricardite", 0.089],
    ["Embolie pulmonaire", 0.078],
    ["Pneumothorax spontané", 0.077]
  ],
  "Processed Diagnosis": "Péricardite"  ← Ground truth (for evaluation)
}
```

**This contains everything you'd extract from a real EHR:**
- Demographics (age, sex)
- Chief complaint (initial evidence)
- Clinical findings (evidences list)
- Differential diagnosis considerations
- True diagnosis (known ground truth)

### Why DDXPlus Works Perfectly for MedRAG

```
MedRAG Pipeline Needs:
├─ Patient demographics     ← ✓ DDXPlus has this
├─ Clinical symptoms        ← ✓ DDXPlus has this
├─ Structured format        ← ✓ DDXPlus is clean JSON
├─ Ground truth labels      ← ✓ DDXPlus has known diagnoses
├─ Multiple similar cases   ← ✓ DDXPlus has 1M+ records (we use 13,230)
└─ Knowledge graph mapping  ← ✓ We have DDXPlus KG

The paper's authors used DDXPlus for evaluation!
```

---

## Summary

### Your Questions - Final Visual Answer

**Q1: Testing subset - need all embeddings?**

```
Test 1 patient  ────►  Search 11,760 training embeddings  ────►  Find best match
Test 10 patients ───►  Search 11,760 training embeddings  ────►  Find best matches
Test 1,470 patients ►  Search 11,760 training embeddings  ────►  Find best matches

                        ↑ SAME DATABASE EVERY TIME ↑
```

**Answer: YES, always need all 11,760 training embeddings**

**Q2: What are embeddings for?**

```
Text (symptoms) ──► Embedding (numbers) ──► Similarity Search ──► Find similar cases
                    [0.234, -0.567, ...]
```

**Answer: Convert clinical text to numbers for semantic similarity search**

**Q3: DDXPlus not real EHR - will it work?**

```
DDXPlus = Synthetic EHR = Perfect for Research
├─ Simulates real patient records
├─ Contains: age, sex, symptoms, diagnosis
├─ Used by MedRAG paper authors
└─ Better than real EHR for research (clean, labeled, public)
```

**Answer: YES, DDXPlus IS synthetic EHR data - pipeline works perfectly**

---

## One-Sentence Summary

**You need all 11,760 training embeddings (generated once, $20) to enable semantic search for ANY number of test patients (cheap per patient, $0.02), and DDXPlus synthetic EHR data works perfectly because it simulates real patient records in structured format.**

---

**Next Step:** Set up your API keys and run the first test! See `PHASE3_EXECUTION_GUIDE.md`

