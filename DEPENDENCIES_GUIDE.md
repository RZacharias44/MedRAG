# Dependencies Guide - What You Need to Install

## 🎯 Quick Answer

You need ALL packages from `requirements.txt` (31 packages total).

Currently installed: Only 3 packages (numpy, pandas, matplotlib) ❌  
Need to install: 28 more packages ⚠️

---

## 📦 Critical Missing Packages

### Must-Have for Embeddings
```
❌ python-dotenv    # Load .env file
❌ openai           # OpenAI API (embeddings & LLM)
❌ faiss-cpu        # Similarity search (FAISS)
❌ nltk             # Text preprocessing
❌ scikit-learn     # Cosine similarity
❌ scipy            # Scientific computing
```

### Must-Have for Pipeline
```
❌ tqdm             # Progress bars
❌ transformers     # Hugging Face models (optional)
❌ huggingface-hub  # HF API (optional)
❌ torch            # PyTorch (for some models)
❌ networkx         # Graph operations (for KG)
```

### Supporting Packages
```
❌ openpyxl         # Read Excel files (KG)
❌ regex            # Text processing
❌ click            # CLI utilities
❌ joblib           # Caching
... and more
```

---

## 🚀 How to Install (Two Options)

### Option 1: Using uv + requirements.txt (Recommended)

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Install all packages from requirements.txt
uv pip install -r requirements.txt

# This installs all 31 packages including:
# - python-dotenv (for .env files)
# - openai (for API calls)
# - faiss-cpu (for retrieval)
# - pandas, numpy, nltk, etc.
```

**Time:** 2-5 minutes  
**Size:** ~2-3 GB (includes PyTorch)

### Option 2: Update pyproject.toml (Better for uv)

The current `pyproject.toml` only lists 3 packages. Let me create a complete one:

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Backup current file
cp pyproject.toml pyproject.toml.backup

# I'll create a complete pyproject.toml for you
```

---

## 📋 Complete List of Required Packages

### Core Dependencies (from requirements.txt)
```
1.  python-dotenv==1.0.1       # .env file loading ✅ CRITICAL
2.  openai==1.51.0             # OpenAI API ✅ CRITICAL
3.  faiss-cpu==1.8.0.post1     # FAISS search ✅ CRITICAL
4.  pandas==2.2.3              # Data handling ✅ Already installed
5.  numpy==1.26.4              # Arrays ✅ Already installed
6.  scikit-learn==1.5.2        # ML utilities ✅ CRITICAL
7.  scipy==1.14.1              # Scientific computing ✅ CRITICAL
8.  nltk==3.9.1                # NLP preprocessing ✅ CRITICAL
9.  tqdm==4.66.5               # Progress bars
10. click==8.1.7               # CLI utilities
11. joblib==1.4.2              # Caching
12. regex==2024.9.11           # Text processing
13. protobuf==5.28.2           # Data serialization
14. PyYAML==6.0.2              # YAML parsing
15. packaging==24.1            # Version handling
16. python-dateutil==2.9.0     # Date utilities
17. pytz==2024.2               # Timezone handling
```

### Optional but Included
```
18. torch==2.4.1               # PyTorch (1.5GB!)
19. transformers==4.45.1       # Hugging Face models
20. huggingface-hub==0.25.1    # HF API
21. accelerate==0.34.2         # Model acceleration
22. safetensors==0.4.5         # Tensor storage
23. tokenizers==0.20.0         # Fast tokenizers
24. sentencepiece==0.2.0       # Tokenization
25. spacy==3.7.5               # NLP library
26. rank-bm25==0.2.2           # BM25 retrieval
27. psutil==6.0.0              # System utilities
28. anyio==4.6.0               # Async I/O
29. httpx==0.27.2              # HTTP client
30. wheel==0.44.0              # Package building
```

---

## ⚡ Quick Install Command

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Install everything
uv pip install -r requirements.txt

# Download NLTK data (required)
uv run python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Verify critical packages
uv run python3 -c "
import openai
import faiss
import nltk
import sklearn
import pandas
from dotenv import load_dotenv
print('✅ All critical packages installed!')
"
```

---

## 🔍 Why Each Package is Needed

### python-dotenv
```python
# Loads .env file with API keys
from dotenv import load_dotenv
load_dotenv()
```
**Used in:** `authentication.py`  
**Critical:** YES - Can't load API keys without it

### openai
```python
# Creates embeddings and calls LLM
import openai
client = openai.OpenAI(api_key=api_key)
response = client.embeddings.create(...)
```
**Used in:** `main_MedRAG.py`, `KG_Retrieve.py`  
**Critical:** YES - Core functionality

### faiss-cpu
```python
# Fast similarity search
import faiss
index = faiss.IndexFlatIP(dimensions)
index.search(query_embedding, k)
```
**Used in:** `main_MedRAG.py`  
**Critical:** YES - Retrieval component

### nltk
```python
# Text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```
**Used in:** `KG_Retrieve.py`  
**Critical:** YES - KG symptom matching

### scikit-learn
```python
# Cosine similarity for KG matching
from sklearn.metrics.pairwise import cosine_similarity
```
**Used in:** `KG_Retrieve.py`  
**Critical:** YES - KG augmentation

### pandas
```python
# Data manipulation
import pandas as pd
df = pd.read_csv(...)
df = pd.read_excel(...)
```
**Used in:** Everywhere  
**Critical:** YES - Already installed ✅

### numpy
```python
# Array operations
import numpy as np
embeddings = np.array(...)
np.save(path, embeddings)
```
**Used in:** Everywhere  
**Critical:** YES - Already installed ✅

### torch
```python
# PyTorch (for Hugging Face models)
import torch
```
**Used in:** If using open-source LLMs  
**Critical:** NO - Only if using Llama/Mistral

---

## 📦 Installation Sizes

```
Total download: ~2.5 GB
Total installed: ~3.5 GB

Breakdown:
- PyTorch:           ~1.5 GB (largest)
- Transformers:      ~500 MB
- FAISS:             ~50 MB
- OpenAI:            ~10 MB
- Other packages:    ~440 MB
```

**Disk space needed:** ~4 GB free

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'dotenv'"

```bash
# Solution:
uv pip install python-dotenv
```

### Issue: "ModuleNotFoundError: No module named 'openai'"

```bash
# Solution:
uv pip install openai
```

### Issue: "ModuleNotFoundError: No module named 'faiss'"

```bash
# Solution:
uv pip install faiss-cpu
```

### Issue: All packages fail to install

```bash
# Solution: Install one by one to find the problem
uv pip install python-dotenv
uv pip install openai
uv pip install faiss-cpu
uv pip install pandas
uv pip install scikit-learn
# ... continue with others
```

### Issue: NLTK data not found

```bash
# Solution: Download required NLTK data
uv run python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
print('✅ NLTK data downloaded')
"
```

---

## ✅ Verification Script

After installation, run this to verify everything works:

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

uv run python3 << 'EOF'
print("Checking dependencies...")

# Check each critical package
packages = {
    'python-dotenv': 'dotenv',
    'openai': 'openai',
    'faiss-cpu': 'faiss',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'scipy': 'scipy',
    'nltk': 'nltk',
    'tqdm': 'tqdm',
}

missing = []
for name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name} - MISSING")
        missing.append(name)

if missing:
    print(f"\n❌ Missing {len(missing)} packages:")
    for pkg in missing:
        print(f"   - {pkg}")
    print("\nRun: uv pip install " + " ".join(missing))
else:
    print("\n🎉 All critical dependencies installed!")
EOF
```

---

## 🚀 Complete Setup Sequence

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Step 1: Install all packages
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt

# Step 2: Download NLTK data
echo "📚 Downloading NLTK data..."
uv run python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Step 3: Verify installation
echo "✅ Verifying installation..."
uv run python3 -c "
import openai
import faiss
import nltk
import sklearn
from dotenv import load_dotenv
print('✅ All packages working!')
"

# Step 4: Test API key
echo "🔑 Testing API key..."
uv run python3 -c "
from authentication import api_key
print('API Key loaded:', api_key[:15] + '...' if api_key else 'NOT SET')
"

echo "🎉 Setup complete! Ready to run."
```

---

## 📊 Summary

**Current status:**
- ✅ Installed: 3 packages (numpy, pandas, matplotlib)
- ❌ Missing: 28 packages (including critical ones)

**What you need:**
```bash
# Single command to install everything:
cd "/Users/sunray/Documents/masters thesis/MedRAG"
uv pip install -r requirements.txt
```

**Time:** 2-5 minutes  
**Size:** ~3.5 GB installed  
**Critical packages:**
1. python-dotenv (for .env)
2. openai (for API)
3. faiss-cpu (for FAISS)
4. nltk (for preprocessing)
5. scikit-learn (for similarity)

**After installation:**
```bash
# Download NLTK data
uv run python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Ready to run!
uv run main.py
```

---

**Next:** Run `uv pip install -r requirements.txt` then you're ready for embedding generation! 🚀

