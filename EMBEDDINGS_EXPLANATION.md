# Understanding Embeddings in MedRAG

## Your Questions Answered

### Q1: When testing a small subset, do all training embeddings need to be generated?

**Answer: YES** - All 11,760 training embeddings must be generated, even if you only test 1 patient.

### Q2: What are the embeddings used for?

**Answer:** Embeddings enable **semantic similarity search** for retrieval-augmented generation.

### Q3: We don't have real EHR data - will it still work?

**Answer: YES** - DDXPlus is synthetic EHR data designed for this purpose.

---

## How Embeddings Work in MedRAG

### The Pipeline Flow

```
Test Patient → Embed → Search Training Set → Retrieve Similar Cases → LLM + KG → Diagnosis
     (1)         (2)           (3)                  (4)              (5)        (6)
```

### Detailed Steps

#### Step 1: Load ALL Training Documents (One-Time Setup)

```python
# In main.py (happens before the loop)
documents = []  # All training patient JSON files
for file in os.listdir(ob_path):  # ob_path = './dataset/DDXPlus/train'
    documents.append(os.path.join(ob_path, file))

# Result: documents = [
#   './dataset/DDXPlus/train/participant_1.json',
#   './dataset/DDXPlus/train/participant_2.json',
#   ...
#   './dataset/DDXPlus/train/participant_11760.json'
# ]
```

**Count: 11,760 training documents**

#### Step 2: Generate/Load Training Embeddings (One-Time, Cached)

```python
# Convert ALL training documents to embeddings
document_embeddings = get_embeddings(documents)

# Uses OpenAI text-embedding-3-large
# - Input: Patient JSON (age, sex, symptoms, evidences)
# - Output: 3,072-dimensional vector representing semantic meaning
```

**What gets embedded:**
```json
{
  "Participant No.": 1,
  "Age": 43,
  "Sex": "F",
  "Evidences": "['douleurxx', 'dyspn', 'palpit', ...]",
  "Initial Evidence": "palpit",
  "Processed Diagnosis": "Péricardite"
}
```

**Result:** 
- Shape: `(11760, 3072)` - Matrix of all training embeddings
- Cached in `./Embeddings_saved/CP_DC_embeddings/` (loaded on subsequent runs)
- **Cost: $15-20 USD (one-time)**

#### Step 3: For EACH Test Patient - Embed Query

```python
# Test patient 1
query = json.dumps(test_patient_1)  # Convert to string
query_embedding = get_query_embedding(query)  # Shape: (3072,)
```

**Cost per query: $0.0001 USD**

#### Step 4: FAISS Similarity Search

```python
# Search across ALL 11,760 training embeddings
indices = Faiss(document_embeddings, query_embedding, k=topk)

# Returns: indices of top-k most similar training patients
# Example: topk=1 → indices = [5432]
# Meaning: training patient #5432 is most similar to test patient #1
```

**Why all training embeddings are needed:**
- FAISS builds an index over ALL 11,760 embeddings
- Even for 1 test patient, it searches the entire training set
- This is the "Retrieval" in Retrieval-Augmented Generation

**Search method:** Inner Product (IndexFlatIP)
- Higher inner product = more similar
- Fast: searches 11,760 embeddings in milliseconds

#### Step 5: Retrieve Similar Cases

```python
# Get the actual patient data for similar cases
retrieved_documents = [documents[i] for i in indices[0]]

# Load the JSON files
for doc_path in retrieved_documents:
    with open(doc_path, 'r') as f:
        similar_patient = json.load(f)
```

**Purpose:** Provide context to the LLM
- Shows similar patient cases with known diagnoses
- Helps LLM understand patterns
- Improves diagnostic accuracy

#### Step 6: KG-Elicited Reasoning + LLM

```python
# Get relevant info from Knowledge Graph
kg_info = get_additional_info_from_level_2(participant_no, kg_path, top_n, match_n)

# Generate diagnosis using LLM
prompt = f"""
Query: {test_patient}
Retrieved Similar Cases: {similar_cases}
Knowledge Graph Info: {kg_info}
"""

diagnosis = llm.generate(prompt)  # GPT-4o
```

---

## Why You Need All Training Embeddings

### Scenario: Testing 1 Patient vs 1,470 Patients

| Scenario | Training Embeddings Needed | Why |
|----------|---------------------------|-----|
| Test 1 patient | All 11,760 | FAISS searches entire training set |
| Test 10 patients | All 11,760 | Each test query searches full training set |
| Test 1,470 patients | All 11,760 | Every query needs access to all training data |

**It's like a database:**
- Training embeddings = Database with 11,760 records
- Test query = SQL query searching the database
- You can't query 1 row without having the whole database

### Embedding Types in MedRAG

```
1. TRAINING DOCUMENT EMBEDDINGS (11,760)
   - Purpose: Enable similarity search
   - Generated: Once (first run)
   - Cached: ./Embeddings_saved/CP_DC_embeddings/
   - Cost: $15-20 (one-time)
   
2. TEST QUERY EMBEDDINGS (1,470 total, or 1 if testing subset)
   - Purpose: Search against training embeddings
   - Generated: For each test patient
   - Not cached: Generated fresh each time
   - Cost: $0.0001 per patient
   
3. KG SYMPTOM EMBEDDINGS (~500)
   - Purpose: Match patient symptoms to KG nodes
   - Generated: Once (first run)
   - Cached: ./Embeddings_saved/DDXPlus_KG_embeddings/
   - Cost: $0.10 (one-time)
```

---

## About the "EHR" Data (DDXPlus)

### You're Correct - This is Synthetic Data

**DDXPlus is NOT real patient data.** It's synthetic, which is actually **perfect** for research:

| Aspect | Real EHR | DDXPlus (Synthetic) |
|--------|----------|---------------------|
| Privacy | High risk | No privacy concerns ✓ |
| Availability | Restricted | Public dataset ✓ |
| Structure | Messy, inconsistent | Clean, structured ✓ |
| Ground truth | Often uncertain | Known diagnoses ✓ |
| Research use | Difficult | Ideal ✓ |

### What DDXPlus Contains (Simulated EHR Fields)

```json
{
  "Participant No.": 1,
  "Processed Diagnosis": "Péricardite",  // Ground truth diagnosis
  "Age": 43,
  "Sex": "F",
  "Evidences": "['douleurxx', 'douleurxx_carac_@_un_coup_de_couteau', ...]",
  "Initial Evidence": "palpit",
  "Differential Diagnosis": "[[Péricardite, 0.089], [Embolie pulmonaire, 0.078], ...]"
}
```

**This simulates what you'd extract from real EHR:**
- Patient demographics (age, sex)
- Chief complaint (initial evidence)
- Symptoms/findings (evidences)
- Differential diagnosis list
- True diagnosis (ground truth)

### Will the Pipeline Work? YES!

The pipeline treats these JSON files as "EHR records." From the code's perspective:
- ✅ Each JSON = 1 patient record
- ✅ Contains clinical information (symptoms, demographics)
- ✅ Has diagnosis label (for evaluation)
- ✅ Structured format enables embedding

**The MedRAG paper itself used DDXPlus** alongside their private CPDD dataset. So your setup is exactly what the authors intended!

---

## Cost Breakdown

### First Run (Generates All Embeddings)

```
Training embeddings (11,760):     $15-20
KG symptom embeddings (~500):     $0.10
Test query (1 patient):           $0.0001
LLM diagnosis (1 patient):        $0.01-0.02
─────────────────────────────────────────
Total (first run, 1 test patient): ~$20
```

### Testing Small Subset (10 patients, embeddings cached)

```
Training embeddings:              $0 (cached)
KG embeddings:                    $0 (cached)
Test queries (10 patients):       $0.001
LLM diagnoses (10 patients):      $0.10-0.20
─────────────────────────────────────────
Total (10 patients, cached):      ~$0.20
```

### Full Test Set (1,470 patients, embeddings cached)

```
Training embeddings:              $0 (cached)
KG embeddings:                    $0 (cached)
Test queries (1,470 patients):    $0.15
LLM diagnoses (1,470 patients):   $15-30
─────────────────────────────────────────
Total (1,470 patients, cached):   ~$30
```

**Key Point:** The expensive part ($15-20) happens ONCE. After that, testing is cheap!

---

## Optimization Strategies

### Strategy 1: Test with Pre-Generated Embeddings

If you want to test just 1 patient without generating all embeddings:

**Problem:** Can't do it - FAISS needs the full training set.

**Alternative:** Use a smaller training subset for testing:
```python
# Modify main.py to load only 100 training documents
documents = documents[:100]  # Use first 100 only
document_embeddings = get_embeddings(documents[:100])
```

**Trade-off:** Less accurate retrieval (smaller reference set)

### Strategy 2: Share Embeddings

If someone else has already run MedRAG on DDXPlus:
```bash
# Copy their cached embeddings
cp -r their_embeddings/* ./Embeddings_saved/
```

**Result:** Skip $15-20 embedding generation cost

### Strategy 3: Start Small, Scale Up

```python
# Day 1: Test 1 patient (generates all embeddings)
samplerange = range(1, 2)      # Cost: ~$20
uv run main.py

# Day 2: Test 10 more (embeddings cached)
samplerange = range(2, 12)     # Cost: ~$0.20
uv run main.py

# Day 3: Full test set (embeddings cached)
samplerange = range(1, 1471)   # Cost: ~$30
uv run main.py
```

---

## Summary

### Your Questions - Final Answers

**Q1: Do embeddings need to be generated for the whole dataset when testing a subset?**

✅ **YES** - All 11,760 training embeddings must be generated because:
- FAISS searches the entire training set for each test query
- You can't search a database without having the database
- This is fundamental to Retrieval-Augmented Generation
- **BUT:** This happens ONCE and is cached - subsequent runs are cheap

**Q2: What are embeddings used for?**

✅ Embeddings enable **semantic similarity search**:
- Convert patient records to numerical vectors (3,072 dimensions)
- Similar patients have similar vectors (high inner product)
- FAISS quickly finds most similar training cases
- Retrieved cases provide context to LLM for better diagnosis

**Q3: We don't have EHR data - will it work?**

✅ **YES** - DDXPlus IS EHR data (synthetic):
- Simulates real clinical records with symptoms, demographics, diagnoses
- The MedRAG paper used DDXPlus alongside real data
- Synthetic data is better for research (privacy-safe, clean, labeled)
- The pipeline treats JSON files as "EHR records" and works perfectly

### The Big Picture

```
One-Time Setup (First Run):
├─ Generate 11,760 training embeddings → $15-20 (cached forever)
├─ Generate ~500 KG embeddings → $0.10 (cached forever)
└─ Total: ~$20 (never pay again)

Testing (Any Number of Patients):
├─ Generate query embedding → $0.0001 per patient
├─ Search FAISS (uses cached embeddings) → FREE
├─ LLM diagnosis → $0.01-0.02 per patient
└─ Total: ~$0.02 per patient

Full Experiment (1,470 patients):
└─ Total: ~$50 (first run) or ~$30 (cached embeddings)
```

**Bottom Line:** The initial embedding generation is the "database setup" cost. Once done, you can test as many patients as you want cheaply. This is the standard approach for RAG systems!

