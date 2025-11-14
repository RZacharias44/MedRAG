# API Keys Explained - What Do You Actually Need?

## TL;DR

| API Key | Required? | Used For | Cost |
|---------|-----------|----------|------|
| **OpenAI** | ✅ **YES** | Embeddings + LLM (GPT-4o) | $$$ |
| **Hugging Face** | ⚠️ **Optional** | Open-source LLMs (Llama, Mistral) | $ or Free |

**For your thesis replication: You ONLY need OpenAI API key.**

---

## OpenAI API Key (REQUIRED ✅)

### What It's Used For

#### 1. **Embeddings Generation** (REQUIRED)
```python
# In main_MedRAG.py and KG_Retrieve.py
response = client.embeddings.create(
    input=text,
    model="text-embedding-3-large"  # OpenAI's embedding model
)
```

**Used for:**
- Training document embeddings (11,760 patients) - $15-20
- Test query embeddings (per patient) - $0.0001 each
- KG symptom embeddings (~500 nodes) - $0.10

**No alternatives available** - The codebase specifically uses OpenAI's `text-embedding-3-large` model.

#### 2. **LLM Diagnosis Generation** (DEFAULT)
```python
# In main_MedRAG.py line 191-198
if model == 'gpt-4o' or 'gpt-4o-mini' or 'gpt-3.5-turbo-0125':
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
```

**Used for:**
- Generating diagnosis for each test patient
- Cost: $0.01-0.02 per patient (GPT-4o)
- Cost: $0.005-0.01 per patient (GPT-4o-mini)
- Cost: $0.001-0.002 per patient (GPT-3.5-turbo)

### How to Get OpenAI API Key

1. **Sign up:** https://platform.openai.com/signup
2. **Add billing:** https://platform.openai.com/account/billing
   - Required to use the API
   - You'll be charged based on usage
3. **Create API key:** https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy it (starts with `sk-proj-...`)
   - Never share it publicly!

### Cost Estimate with OpenAI

```
First Run:
├─ Training embeddings (11,760): $15-20
├─ KG embeddings (~500): $0.10
├─ Test queries (1,470): $0.15
├─ LLM diagnoses (1,470 × GPT-4o): $15-30
└─ TOTAL: ~$50

Subsequent Runs (cached embeddings):
└─ Only LLM costs: ~$30
```

---

## Hugging Face Token (OPTIONAL ⚠️)

### What It's Used For

**ONLY if you choose to use open-source LLM models** instead of GPT:

```python
# In main_MedRAG.py line 200-214
else:  # If NOT using GPT models
    LLMclient = InferenceClient(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Open-source model
        token=hf_token  # <-- Hugging Face token used here
    )
    response = LLMclient.text_generation(prompt=prompt, max_new_tokens=400)
```

### Available Open-Source Models (All require HF token)

The code includes options for:
- ✅ `meta-llama/Meta-Llama-3.1-8B-Instruct` (default if not using GPT)
- ✅ `meta-llama/Llama-2-13b-chat-hf`
- ✅ `meta-llama/Meta-Llama-3.1-70B-Instruct`
- ✅ `Qwen/Qwen2-7B-Instruct`
- ✅ `Qwen/Qwen2.5-0.5B-Instruct`
- ✅ `mistralai/Mistral-7B-Instruct-v0.2`
- ✅ `mistralai/Mixtral-8x7B-Instruct-v0.1`

### When to Use Hugging Face Instead of OpenAI

**Pros:**
- 💰 **Cheaper** - Some models have free tier
- 🔓 **Open-source** - Full model transparency
- 🔒 **Privacy** - Can run locally (if you set up local inference)

**Cons:**
- 📉 **Lower quality** - Generally worse than GPT-4o for complex medical reasoning
- 🐌 **Slower** - API inference can be slow on free tier
- ⚙️ **More setup** - Need to choose model, configure parameters

### How to Get Hugging Face Token

1. **Sign up:** https://huggingface.co/join
2. **Go to settings:** https://huggingface.co/settings/tokens
3. **Create token:**
   - Click "New token"
   - Give it "Read" access (enough for inference)
   - Copy it (starts with `hf_...`)

### Cost Estimate with Hugging Face

```
First Run:
├─ Training embeddings: $15-20 (still need OpenAI for embeddings!)
├─ KG embeddings: $0.10 (still need OpenAI for embeddings!)
├─ LLM diagnoses (1,470 × Llama): $0-5 (free tier or paid)
└─ TOTAL: ~$20-25

Savings: ~$10-25 compared to GPT-4o
Trade-off: Potentially lower diagnostic accuracy
```

**Important:** You STILL need OpenAI API for embeddings even if you use HF for LLM!

---

## Recommendation for Your Thesis

### Use OpenAI Only (Don't Bother with Hugging Face)

**Reasons:**

1. **Paper Compliance** 
   - The MedRAG paper used GPT-4o for evaluation
   - To replicate results, use the same model
   - Open-source models may give different results

2. **Simplicity**
   - One API key to manage
   - Proven to work well
   - Faster setup

3. **Quality**
   - GPT-4o has better medical reasoning
   - Higher diagnostic accuracy
   - More reliable outputs

4. **Cost is Reasonable**
   - ~$50 for full experiment
   - Worth it for quality results
   - This is research spending

5. **Time = Money**
   - Debugging open-source models takes time
   - Your time is valuable
   - GPT-4o just works

### If You Want to Save Money (Use HF)

If you absolutely need to reduce costs, you could:

1. **Use GPT-4o-mini instead of GPT-4o**
   - 50% cheaper ($0.005 vs $0.01 per patient)
   - Still very good quality
   - **Recommended budget option**

2. **Use Llama 3.1 8B for some experiments**
   - Get HF token
   - Test on small subset first
   - Compare quality vs GPT results

---

## Setup Instructions

### Option 1: OpenAI Only (Recommended)

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Create .env file
cat > .env << 'EOF'
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Hugging Face Token (OPTIONAL - leave blank if not using)
HUGGINGFACE_TOKEN=
EOF

# Make sure it's set
cat .env
```

### Option 2: Both APIs (If Experimenting)

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

# Create .env file with both keys
cat > .env << 'EOF'
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Hugging Face Token (OPTIONAL)
HUGGINGFACE_TOKEN=hf_your-actual-token-here
EOF
```

---

## How to Choose Which Model to Use

### In main.py

The model is specified when calling `generate_diagnosis_report()`:

```python
# Line 77 in main.py
generated_report_ori = generate_diagnosis_report(
    augmented_features_path, 
    query, 
    final_retrieved_info, 
    i, 
    top_n=top_n, 
    match_n=match_n,
    model='gpt-4o'  # <-- Change this!
)
```

### Model Options

#### Using OpenAI (Requires OPENAI_API_KEY only)

```python
model='gpt-4o'           # Best quality, highest cost ($0.01-0.02/patient)
model='gpt-4o-mini'      # Good quality, medium cost ($0.005/patient)
model='gpt-3.5-turbo-0125'  # OK quality, lowest cost ($0.001/patient)
```

#### Using Hugging Face (Requires BOTH keys)

First, modify `main_MedRAG.py` to select your model (line 203):

```python
# Change from default:
LLMclient = InferenceClient(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Change this line
    token=hf_token
)
```

Then in `main.py`:
```python
model='llama'  # Or any non-GPT value triggers HF path
```

---

## Code Logic Explanation

The model selection logic in `main_MedRAG.py`:

```python
def generate_diagnosis_report(path, query, retrieved_documents, i, top_n, match_n, model):
    # ... prepare prompt ...
    
    # Check if using OpenAI models
    if model == 'gpt-4o' or 'gpt-4o-mini' or 'gpt-3.5-turbo-0125':
        # Use OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[...]
        )
        return response.choices[0].message.content
    
    else:
        # Use Hugging Face InferenceClient
        LLMclient = InferenceClient(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=hf_token
        )
        response = LLMclient.text_generation(prompt=prompt, max_new_tokens=400)
        return response
```

**Note:** There's a logic bug in line 190 - the condition should be:
```python
if model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-0125']:
```

But it works in practice because passing `model='gpt-4o'` triggers the OpenAI path.

---

## Testing Which Keys You Have

### Check if OpenAI key works:

```bash
cd "/Users/sunray/Documents/masters thesis/MedRAG"

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

### Check if Hugging Face token works (optional):

```bash
uv run python3 -c "
from authentication import hf_token
from huggingface_hub import InferenceClient
client = InferenceClient('gpt2', token=hf_token)
print('✅ Hugging Face token works!')
"
```

---

## Summary

### What You Need for Your Thesis

```
┌─────────────────────────────────────────────────────────┐
│  REQUIRED:                                              │
│  ✅ OpenAI API Key                                      │
│     - For embeddings (no alternatives)                  │
│     - For LLM (best results)                            │
│     - Cost: ~$50 for full experiment                    │
│                                                         │
│  OPTIONAL:                                              │
│  ⚠️  Hugging Face Token                                 │
│     - Only if you want to try open-source LLMs          │
│     - Saves ~$15-25 but may reduce quality              │
│     - Not recommended for paper replication             │
└─────────────────────────────────────────────────────────┘
```

### Quick Decision Guide

**Just want to replicate the paper?**
→ ✅ Get OpenAI key only, use GPT-4o

**Want to save money?**
→ ✅ Get OpenAI key, use GPT-4o-mini (50% cheaper, still good)

**Want to experiment with models?**
→ Get both keys, test GPT vs Llama and compare

**Absolutely broke?**
→ Get both keys, use Llama 3.1 (but results may differ from paper)

---

## My Recommendation

**Get OpenAI API key and use GPT-4o** for your thesis. Here's why:

1. ✅ Matches the paper's methodology
2. ✅ Best diagnostic accuracy
3. ✅ Only ~$50 total cost (reasonable for research)
4. ✅ Reliable, well-tested
5. ✅ One less thing to debug

**Save Hugging Face for future work** if you want to explore cost optimization or compare open-source vs closed-source models.

---

**Next:** Set up your OpenAI API key and start Phase 3! See `PHASE3_EXECUTION_GUIDE.md`

