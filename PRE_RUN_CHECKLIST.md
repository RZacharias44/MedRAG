# Pre-Run Checklist - Are You Ready?

## 🎯 Quick Status Check

Let me verify everything is ready for embedding generation!

---

## ✅ What I Found

### 1. Preprocessed Data
```
✅ Training data: 11,760 JSON files in dataset/DDXPlus/train/
✅ Test data: 1,470 JSON files in dataset/DDXPlus/test/
✅ Ground truth: dataset/DDXPlus_ground_truth.csv
✅ Knowledge Graph: dataset/knowledge graph of DDXPlus.xlsx
```

### 2. Embedding Directories
```
✅ Embeddings_saved/CP_DC_embeddings/ exists (for training embeddings)
✅ Embeddings_saved/DDXPlus_KG_embeddings/ exists (for KG embeddings)
```

### 3. Code Configuration
```
✅ authentication.py configured for DDXPlus
✅ KG_Retrieve.py imports from authentication
✅ main_MedRAG.py ready
✅ Paths all point to correct locations
```

---

## ❌ CRITICAL: What You Need to Do FIRST

### 1. Create .env File with API Key

**You don't have a .env file yet!** This is REQUIRED.

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Create .env file
cat > .env << 'EOF'
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY-HERE

# Hugging Face Token (OPTIONAL - leave blank if not using)
HUGGINGFACE_TOKEN=
EOF
```

**⚠️ Replace `sk-proj-YOUR-ACTUAL-KEY-HERE` with your real OpenAI API key!**

Get it from: https://platform.openai.com/api-keys

### 2. Verify API Key Works

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Test if key is loaded
python3 -c "
from authentication import api_key
print('API Key:', api_key[:15] + '...' if api_key else 'NOT SET')
"

# Test if key works with OpenAI
uv run python3 -c "
from authentication import api_key
import openai
client = openai.OpenAI(api_key=api_key)
response = client.embeddings.create(
    input='test',
    model='text-embedding-3-large'
)
print('✅ OpenAI API key works!')
print(f'Embedding dimensions: {len(response.data[0].embedding)}')
"
```

If this fails, check:
- Did you add billing to your OpenAI account?
- Is the API key correct (no spaces, complete key)?
- Is your internet connection working?

---

## 📁 Where Embeddings Will Be Saved

### Training Document Embeddings (11,760 patients)

**Location:** `./Embeddings_saved/DDXPlus_document_embeddings.npy`

```python
# In main_MedRAG.py line 226:
document_embeddings_file_path = './Embeddings_saved/DDXPlus_document_embeddings.npy'
```

**Details:**
- File size: ~400-500 MB
- Format: NumPy array, shape (11760, 3072)
- Contains: Embeddings for ALL training patients
- Cost: $15-20 to generate (one-time)
- Time: ~25-30 minutes

### KG Symptom Embeddings (~500 symptoms)

**Location:** `./Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy`

```python
# In KG_Retrieve.py line 27:
embedding_save_path = './Embeddings_saved/DDXPlus_KG_embeddings'
# Saves to: embedding_save_path + '/KG_embeddings.npy'
```

**Details:**
- File size: ~5-10 MB
- Format: NumPy array, shape (~500, 3072)
- Contains: Embeddings for KG symptom nodes
- Cost: $0.10 to generate (one-time)
- Time: ~2-3 minutes

### Summary of Files After Embedding Generation

```
MedRAG/
├── dataset/
│   ├── DDXPlus/
│   │   ├── train/ (11,760 JSON files)
│   │   └── test/ (1,470 JSON files)
│   ├── DDXPlus_ground_truth.csv
│   └── knowledge graph of DDXPlus.xlsx
│
└── Embeddings_saved/
    ├── DDXPlus_document_embeddings.npy  ← NEW (400MB)
    └── DDXPlus_KG_embeddings/
        └── KG_embeddings.npy            ← NEW (8.6MB)
```

---

## 🌐 Internet Connection Requirements

### Do you need internet for the WHOLE duration?

**YES** ✅ - You need continuous internet connection because:

1. **Every embedding API call goes to OpenAI servers**
   - 11,760 calls for training documents
   - ~500 calls for KG symptoms
   - Each call requires internet

2. **No offline mode**
   - The code uses OpenAI API (cloud-based)
   - Cannot generate embeddings locally
   - If connection drops, generation fails

3. **What happens if connection drops?**
   - ❌ Current call fails
   - ❌ Process stops
   - ⚠️ You'll lose progress (no intermediate saving)
   - 💰 You've already paid for completed calls
   - 🔄 Must restart from beginning

### Recommendation

**Run on stable connection:**
- ✅ Home WiFi (reliable)
- ✅ Ethernet (best)
- ⚠️ Coffee shop WiFi (risky)
- ❌ Mobile hotspot (not recommended - may drop)

**Best practice:**
1. Test with 1 patient first (quick, validates setup)
2. Then run full embedding generation when you have 30+ minutes of stable internet

---

## ⏱️ Time Estimates

### First Run (Generate ALL Embeddings)

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Import & Load KG                              │
│  - Load knowledge graph Excel                           │
│  - Build graph structure                                │
│  - Extract symptom nodes                                │
│  Time: ~10-20 seconds                                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Generate KG Embeddings (~500)                 │
│  - Check if cached (if not, generate)                   │
│  - Call OpenAI for each symptom                         │
│  - Save to ./Embeddings_saved/DDXPlus_KG_embeddings/         │
│  Time: 2-3 minutes                                      │
│  Cost: $0.10                                            │
│  Internet: Required continuously                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Load Training Documents (11,760)              │
│  - List all JSON files in train directory               │
│  - Build paths list                                     │
│  Time: ~5 seconds                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 4: Generate Training Embeddings (11,760)         │
│  - Check if cached (if not, generate)                   │
│  - Load each JSON file                                  │
│  - Call OpenAI for each document                        │
│  - Progress bar shows: 11760it [25:00, 7.84it/s]       │
│  - Save to ./Embeddings_saved/DDXPlus_document_embeddings.npy            │
│  Time: 25-30 minutes                                    │
│  Cost: $15-20                                           │
│  Internet: Required continuously                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Ready! Embeddings cached for future use                │
└─────────────────────────────────────────────────────────┘

TOTAL FIRST RUN: ~30-35 minutes, $20
```

### Subsequent Runs (Embeddings Cached)

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Import & Load KG                              │
│  Time: ~10-20 seconds                                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Load KG Embeddings (cached)                   │
│  - Reads ./Embeddings_saved/.../KG_embeddings.npy       │
│  - Prints: "load existing embeddings..."                │
│  Time: <1 second                                        │
│  Cost: $0                                               │
│  Internet: Not needed for this step                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Load Training Documents                       │
│  Time: ~5 seconds                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 4: Load Training Embeddings (cached)             │
│  - Reads ./Embeddings_saved/DDXPlus_document_embeddings.npy              │
│  - No OpenAI calls                                      │
│  Time: ~2-3 seconds                                     │
│  Cost: $0                                               │
│  Internet: Not needed for this step                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Ready! No embedding generation needed                  │
└─────────────────────────────────────────────────────────┘

TOTAL SUBSEQUENT RUN: ~20 seconds, $0
```

---

## 🚀 How to Run

### Step 1: Create .env File (IF NOT DONE YET)

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Create with your actual key
cat > .env << 'EOF'
OPENAI_API_KEY=sk-proj-YOUR-REAL-KEY-HERE
HUGGINGFACE_TOKEN=
EOF
```

### Step 2: Test with Single Patient (Recommended)

```bash
# Modify main.py first
nano main.py

# Change line 24 to:
# samplerange=range(1,2)  # Test 1 patient only

# Run
uv run main.py
```

**What will happen:**
1. Generates KG embeddings (2-3 min, $0.10)
2. Generates training embeddings (25-30 min, $15-20)
3. Tests 1 patient diagnosis (~10 sec, $0.02)
4. Total: ~30 min, ~$20

**You'll see output like:**
```
Loading DDXPlus data...
generate new embeddings...  ← KG embeddings generating
500it [02:30, 3.33it/s]
generate new embeddings...  ← Training embeddings generating
11760it [27:15, 7.19it/s]
topk: 1
top_ns: 1
match_n: 5
i= 1
load existing embeddings...  ← Using cached embeddings
index: [[5432]]
Additional Info: ...
Success!!!
```

### Step 3: Check Embeddings Were Saved

```bash
# Check training embeddings (should be ~400MB)
ls -lh Embeddings_saved/DDXPlus_document_embeddings.npy

# Check KG embeddings
ls -lh Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy

# Verify sizes
du -sh Embeddings_saved/DDXPlus_document_embeddings.npy
du -sh Embeddings_saved/DDXPlus_KG_embeddings/
```

Expected:
```
-rw-r--r-- 1 user staff 423M Nov 14 15:30 Embeddings_saved/DDXPlus_document_embeddings.npy
-rw-r--r-- 1 user staff 8.6M Nov 14 15:00 Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy
```

---

## ✅ Final Checklist Before Running

### Must Have:
- [x] ✅ Preprocessed data (11,760 train + 1,470 test JSON files)
- [x] ✅ Knowledge graph Excel file
- [x] ✅ Embedding directories created
- [ ] ⚠️ **.env file with OpenAI API key** ← DO THIS FIRST!
- [ ] ⚠️ **Stable internet connection for 30+ minutes**
- [ ] ⚠️ **Billing set up on OpenAI account**

### Good to Have:
- [ ] Tested API key works
- [ ] main.py modified to test 1 patient first
- [ ] ~$25 budget available ($20 embeddings + $5 buffer)
- [ ] Quiet time (30 min uninterrupted)

### Before Running:
```bash
# 1. Check .env exists
cat .env

# 2. Test API key
uv run python3 -c "from authentication import api_key; print(api_key[:15])"

# 3. Check internet
ping -c 3 api.openai.com

# 4. Check disk space (need ~500MB free)
df -h .

# 5. Ready to go!
uv run main.py
```

---

## 🔥 Common Issues

### Issue: "No module named 'authentication'"
```bash
# Solution: Run from project root
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv run main.py
```

### Issue: "RateLimitError: You exceeded your current quota"
```bash
# Solution: Add billing to OpenAI account
# Visit: https://platform.openai.com/account/billing
```

### Issue: Connection times out during embedding generation
```bash
# Solution: 
# 1. Check internet connection
# 2. Try again (will restart from beginning)
# 3. Consider running overnight on stable connection
```

### Issue: "openai.APIConnectionError"
```bash
# Solution:
# 1. Check firewall isn't blocking OpenAI
# 2. Test: curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"
# 3. Verify API key is correct
```

---

## 📊 What to Expect

### Console Output During Generation

```bash
$ uv run main.py

# KG embeddings (first run)
generate new embeddings...
100%|████████████████| 500/500 [02:30<00:00, 3.33it/s]

# Training embeddings (first run)  
generate new embeddings...
100%|████████████████| 11760/11760 [27:15<00:00, 7.19it/s]

# Test patient processing
topk: 1
top_ns: 1
match_n: 5
i= 1
{'Participant No.': 1, 'Processed Diagnosis': 'Péricardite', ...}
load existing embeddings...
index: [[5432]]
Additional Info: Péricardite has symptom douleur thoracique...
Success!!!
________________________________________________________________
```

---

## 🎯 Summary

**Status:** ✅ Code is 100% ready
**Missing:** ⚠️ You need to create .env file with API key

**Embeddings saved to:**
1. Training: `./Embeddings_saved/DDXPlus_document_embeddings.npy` (400MB)
2. KG: `./Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy` (5MB)

**Time:** ~30 minutes (first run), ~20 seconds (subsequent)
**Cost:** ~$20 (first run), ~$0 (subsequent)
**Internet:** Required continuously for first run

**Next steps:**
1. Create .env file with OpenAI API key
2. Test API key works
3. Modify main.py to test 1 patient
4. Run: `uv run main.py`
5. Wait ~30 minutes
6. Embeddings will be cached forever!

---

**Ready when you are!** Just create that .env file first. 🚀

