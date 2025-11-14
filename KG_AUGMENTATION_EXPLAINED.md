# Knowledge Graph Augmentation in MedRAG - Explained

## Your Questions - Quick Answers

### 1. Do we still need to embed symptoms in the Knowledge Graph?

**YES** ✅ - KG symptom embeddings are **essential** for the pipeline to work.

### 2. Was LLM augmentation already applied to the KG?

**NO** ❌ - Augmentation happens **at runtime** for each test patient, not pre-applied to the KG.

---

## Understanding "KG-Elicited Reasoning" (The Core Innovation)

### What is NOT Happening

❌ The Knowledge Graph does NOT contain pre-generated LLM descriptions
❌ The KG is NOT augmented offline before the experiment
❌ You do NOT need to run LLM on the KG first

### What IS Happening

✅ The Knowledge Graph is a **static structure** of medical knowledge
✅ **At runtime**, patient symptoms are matched to KG nodes (using embeddings)
✅ Relevant KG information is **extracted dynamically** for each patient
✅ This KG info is **added to the LLM prompt** alongside retrieved cases
✅ The LLM uses BOTH retrieved cases + KG info to make better diagnoses

---

## The Complete Flow (With KG Augmentation)

### Step-by-Step Process

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: ONE-TIME SETUP - Load Knowledge Graph                  │
└──────────────────────────────────────────────────────────────────┘

Load: knowledge graph of DDXPlus.xlsx
Structure: Subject-Relation-Object triplets

Examples:
┌────────────────────────────────────────────────────────────────┐
│ Subject              Relation        Object                    │
├────────────────────────────────────────────────────────────────┤
│ Pneumothorax spontané  has_symptom   dyspnée                   │
│ Pneumothorax spontané  has_symptom   douleur thoracique        │
│ Pneumonie              has_symptom   dyspnée                   │
│ Pneumonie              has_symptom   fièvre                    │
│ Pneumonie              differs_from  Pneumothorax (has fever)  │
└────────────────────────────────────────────────────────────────┘

Result: knowledge_graph = {
  "Pneumothorax spontané": [
    ("has_symptom", "dyspnée"),
    ("has_symptom", "douleur thoracique"),
    ...
  ],
  "Pneumonie": [
    ("has_symptom", "dyspnée"),
    ("has_symptom", "fièvre"),
    ...
  ]
}


┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: ONE-TIME SETUP - Embed KG Symptom Nodes                │
└──────────────────────────────────────────────────────────────────┘

Extract symptom nodes: ~500 unique symptoms from KG
┌──────────────────────────┐
│ Symptom Nodes:           │
│ - dyspnée                │
│ - douleur thoracique     │
│ - fièvre                 │
│ - toux                   │
│ - ... (~500 total)       │
└──────────────────────────┘
         ↓
Embed each symptom using OpenAI
         ↓
┌─────────────────────────────────────────────────┐
│  KG Symptom Embeddings (cached)                │
│  Shape: (500 symptoms × 3,072 dimensions)       │
│  Saved: ./Embeddings_saved/DDXPlus_KG_embeddings/   │
│  Cost: $0.10 (one-time)                         │
└─────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: FOR EACH TEST PATIENT - Match Symptoms to KG           │
└──────────────────────────────────────────────────────────────────┘

Test Patient Symptoms:
┌──────────────────────────────────┐
│ Patient has:                     │
│ - "dyspnée"                      │
│ - "douleur thoracique aiguë"     │
│ - "début soudain"                │
└──────────────────────────────────┘
         ↓
Embed each patient symptom
         ↓
Search KG symptom embeddings (similarity)
         ↓
┌─────────────────────────────────────────────────┐
│  Top Matching KG Symptoms (top_n=5):           │
│  1. dyspnée (similarity: 0.98)                 │
│  2. douleur thoracique (similarity: 0.92)      │
│  3. toux (similarity: 0.67)                    │
│  4. essoufflement (similarity: 0.65)           │
│  5. oppression thoracique (similarity: 0.63)   │
└─────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: WALK THE GRAPH - Find Related Diseases                 │
└──────────────────────────────────────────────────────────────────┘

For each matched symptom, find diseases in KG:
┌─────────────────────────────────────────────────────────────────┐
│  "dyspnée" is connected to:                                     │
│  - Pneumothorax spontané (via has_symptom)                      │
│  - Pneumonie (via has_symptom)                                  │
│  - Embolie pulmonaire (via has_symptom)                         │
│  - Asthme exacerbé (via has_symptom)                            │
│                                                                 │
│  "douleur thoracique" is connected to:                          │
│  - Pneumothorax spontané (via has_symptom)                      │
│  - Péricardite (via has_symptom)                                │
│  - Possible NSTEMI/STEMI (via has_symptom)                      │
└─────────────────────────────────────────────────────────────────┘
         ↓
Aggregate: Most likely diseases based on symptom overlap
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Candidate Diseases from KG:                                    │
│  1. Pneumothorax spontané (2 symptom matches)                   │
│  2. Pneumonie (1 symptom match)                                 │
│  3. Péricardite (1 symptom match)                               │
└─────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│  STEP 5: EXTRACT KG INFO - Get Diagnostic Differences           │
└──────────────────────────────────────────────────────────────────┘

For candidate diseases, extract KG knowledge:
┌────────────────────────────────────────────────────────────────┐
│  KG Information for Pneumothorax spontané:                     │
│  - has_symptom: dyspnée, douleur thoracique, tympanisme       │
│  - risk_factor: tabagisme, grande taille                       │
│  - differs_from_Pneumonie: no fever, sudden onset             │
│  - diagnosis_test: chest X-ray shows air in pleural space     │
│                                                                │
│  KG Information for Pneumonie:                                 │
│  - has_symptom: dyspnée, toux, fièvre, expectorations         │
│  - differs_from_Pneumothorax: fever present, gradual onset    │
│  - diagnosis_test: chest X-ray shows infiltrate               │
└────────────────────────────────────────────────────────────────┘
         ↓
Format as text for LLM
         ↓
additional_info = "Pneumothorax spontané has symptom dyspnée, 
douleur thoracique; differs from Pneumonie by absence of fever; 
Pneumonie has symptom fièvre, toux; chest X-ray distinguishes..."


┌──────────────────────────────────────────────────────────────────┐
│  STEP 6: AUGMENT LLM PROMPT - Combine Everything                │
└──────────────────────────────────────────────────────────────────┘

Build prompt with THREE sources of information:
┌────────────────────────────────────────────────────────────────┐
│  Prompt to GPT-4o:                                             │
│                                                                │
│  [SYSTEM PROMPT]                                               │
│  You are a medical diagnostic assistant...                     │
│  Must diagnose from these 49 diseases...                       │
│                                                                │
│  [TEST PATIENT - Input]                                        │
│  New Patient:                                                  │
│  - Age: 45, Sex: M                                             │
│  - Symptoms: dyspnée, douleur thoracique aiguë                 │
│  - Onset: sudden                                               │
│                                                                │
│  [RETRIEVED CASES - From RAG]                                  │
│  Similar Patient Found (via FAISS):                            │
│  - Training patient #5432                                      │
│  - Had: dyspnée, douleur thoracique                            │
│  - Diagnosis: Pneumothorax spontané                            │
│  - Treatment: chest tube, oxygen                               │
│                                                                │
│  [KNOWLEDGE GRAPH INFO - KG-Elicited Reasoning] ← THIS IS NEW!│
│  Relevant diagnostic information:                              │
│  - Pneumothorax spontané: dyspnée + chest pain, sudden onset  │
│  - Key difference: NO fever (vs Pneumonie which has fever)    │
│  - Pneumonie: dyspnée + fever + cough                          │
│  - Diagnostic test: Chest X-ray shows air vs infiltrate       │
│                                                                │
│  Now diagnose this patient.                                    │
└────────────────────────────────────────────────────────────────┘
         ↓
         ↓
┌────────────────────────────────────────────────────────────────┐
│  LLM Output (Diagnosis):                                       │
│                                                                │
│  **Diagnosis**: Pneumothorax spontané                          │
│                                                                │
│  **Reasoning**: The patient presents with sudden-onset         │
│  dyspnée and sharp chest pain. The similar case and KG        │
│  confirm this pattern. Critically, the absence of fever       │
│  (as noted in KG) helps distinguish from Pneumonie.           │
│  Recommend chest X-ray to confirm air in pleural space.       │
└────────────────────────────────────────────────────────────────┘
```

---

## Why KG Embeddings Are Essential

### Without KG Embeddings

```
Patient symptom: "shortness of breath"
                      ↓
                 Can't match to KG
                      ↓
              No KG information retrieved
                      ↓
           LLM only has retrieved cases
                      ↓
            Baseline RAG (less accurate)
```

### With KG Embeddings

```
Patient symptom: "shortness of breath"
                      ↓
           Embed + Search KG embeddings
                      ↓
     Match to KG node: "dyspnée" (similarity: 0.95)
                      ↓
        Walk graph to find related diseases
                      ↓
     Extract diagnostic differences from KG
                      ↓
        Add KG info to LLM prompt
                      ↓
  MedRAG with KG-elicited reasoning (more accurate!)
```

**The embeddings enable semantic matching:**
- "shortness of breath" ≈ "dyspnée" (different languages, same concept)
- "chest pain" ≈ "douleur thoracique"
- "cough" ≈ "toux"

Without embeddings, you'd need exact string matching (very brittle).

---

## What Gets Embedded vs What Gets Augmented

### Pre-Generated (One-Time, Cached)

```
✅ Training document embeddings (11,760)
   - Cost: $15-20
   - Cached: Yes
   - Purpose: Enable FAISS retrieval

✅ KG symptom node embeddings (~500)
   - Cost: $0.10
   - Cached: Yes
   - Purpose: Enable semantic symptom matching

✅ Knowledge graph structure (static)
   - From: Excel file (subject-relation-object)
   - No LLM processing
   - Just loaded into memory
```

### Generated at Runtime (Per Patient)

```
🔄 Test query embedding (per patient)
   - Cost: $0.0001 per patient
   - Cached: No
   - Purpose: Search training embeddings

🔄 KG symptom matching (per patient)
   - Match patient symptoms to KG nodes
   - Walk graph to find diseases
   - Extract relevant KG information

🔄 LLM prompt augmentation (per patient)
   - Combine: patient + retrieved cases + KG info
   - Send to LLM
   - Get diagnosis
   - Cost: $0.01-0.02 per patient
```

---

## The Three Levels of Information

### MedRAG uses THREE sources (not just one):

#### 1. Patient Information (Input)
```
Age: 45
Sex: M  
Symptoms: dyspnée, douleur thoracique
Onset: sudden
```

#### 2. Retrieved Similar Cases (RAG Component)
```
Similar patient #5432:
- Had same symptoms
- Diagnosed with: Pneumothorax spontané
- Treatment worked well
```

#### 3. Knowledge Graph Info (KG-Elicited Reasoning - THE INNOVATION!)
```
From medical knowledge:
- Pneumothorax symptoms: A, B, C
- Differs from Pneumonie by: no fever
- Diagnostic tests: chest X-ray pattern
- Key distinguishing features
```

**This is why MedRAG outperforms baseline RAG!**

Baseline RAG = 1 + 2 only
**MedRAG = 1 + 2 + 3** ← Better accuracy!

---

## Code Walkthrough

### Where KG Embeddings Are Generated

```python
# In KG_Retrieve.py (runs once at import)

# 1. Load KG structure
kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])

# 2. Extract symptom nodes (~500)
symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()

# 3. Embed symptoms (cached)
def get_symptom_embeddings(symptom_nodes, save_path):
    embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("load existing embeddings...")  # ← Cached after first run
        return np.load(embeddings_path)
    else:
        print("generate new embeddings...")    # ← First run only
        symptom_embeddings = []
        for symptom in tqdm(symptom_nodes):
            response = client.embeddings.create(
                input=symptom,
                model="text-embedding-3-large"
            )
            symptom_embeddings.append(response.data[0].embedding)
        np.save(embeddings_path, symptom_embeddings)  # Save for next time
        return np.array(symptom_embeddings)

# This runs at import, creates/loads cache
symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)
```

### Where KG Augmentation Happens (Runtime)

```python
# In main_MedRAG.py, function generate_diagnosis_report()

def generate_diagnosis_report(path, query, retrieved_documents, i, top_n, match_n, model):
    # Get system prompt (with disease list)
    system_prompt = get_system_prompt_for_RAGKG()
    
    # ← KG AUGMENTATION HAPPENS HERE!
    additional_info = get_additional_info_from_level_2(i, path, top_n=top_n, match_n=match_n)
    
    # Build prompt with KG info
    prompt = f"""{query}
    Retrieved Documents: {retrieved_documents}
    Information from knowledge graph: {additional_info}
    Now complete the tasks in that format"""
    
    # Send to LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

### What get_additional_info_from_level_2() Does

```python
def get_additional_info_from_level_2(participant_no, kg_path, top_n, match_n):
    # 1. Match patient symptoms to KG symptom nodes (using embeddings)
    level_2_values = main_get_category_and_level3(match_n, participant_no, top_n)
    
    # 2. For each matched symptom, walk the graph
    for level_2_value in level_2_values:
        # Find diseases connected to this symptom
        relevant_level_3_descriptions = [...]
        
        # 3. Extract KG information for those diseases
        kg_data = pd.read_excel(kg_path, usecols=['subject', 'relation', 'object'])
        related_info = kg_data[kg_data['subject'] == level_3]
        
        # 4. Format as text
        for _, row in related_info.iterrows():
            sentence = f"{subject} {relation} {object}"
            additional_info.append(sentence)
    
    # 5. Return KG info as text to add to prompt
    return ', '.join(additional_info)
```

---

## Cost Breakdown with KG Augmentation

### First Run
```
Training embeddings (11,760):     $15-20  (cached)
KG symptom embeddings (~500):     $0.10   (cached) ← KG COST
Test query (1 patient):           $0.0001
LLM with KG augmentation:         $0.01-0.02
──────────────────────────────────────────
Total first run: ~$20
```

### Subsequent Runs
```
Training embeddings:              $0 (cached)
KG embeddings:                    $0 (cached)
Test query (1 patient):           $0.0001
LLM with KG augmentation:         $0.01-0.02
──────────────────────────────────────────
Total per patient: ~$0.02
```

**The KG embedding cost ($0.10) is negligible compared to training embeddings ($15-20)**

---

## Summary

### Your Questions - Final Answers

**Q1: Do we still need to embed KG symptoms?**

✅ **YES, absolutely essential because:**
- Enables semantic matching of patient symptoms to KG nodes
- Without it, can't leverage the knowledge graph
- This is what makes MedRAG better than baseline RAG
- Cost is tiny ($0.10) and cached forever

**Q2: Was LLM augmentation already applied?**

❌ **NO, augmentation happens at runtime because:**
- KG is just a static graph of medical knowledge
- For EACH patient, symptoms are matched to KG dynamically
- Relevant KG info is extracted on-the-fly
- This personalized info is added to the LLM prompt
- LLM sees: patient + similar cases + relevant KG info

### The Complete Picture

```
One-Time Setup:
├─ Load KG structure (Excel file)
├─ Embed training documents → $15-20
└─ Embed KG symptoms → $0.10
    ↓
    All cached!

For Each Test Patient:
├─ Embed query → $0.0001
├─ Retrieve similar cases (FAISS)
├─ Match symptoms to KG (using cached embeddings) ← KG-elicited reasoning
├─ Extract relevant KG info
├─ Augment LLM prompt with KG info
└─ LLM generates diagnosis → $0.01-0.02

Result: Better accuracy than baseline RAG!
```

### Why This Matters for Your Thesis

**MedRAG's Innovation = RAG + Knowledge Graph**

- Baseline RAG: Patient → Retrieve cases → LLM
- **MedRAG: Patient → Retrieve cases + KG reasoning → LLM**

The KG embeddings enable the "+  KG reasoning" part, which is the paper's main contribution!

---

**Next:** Set up your API key and run the pipeline to see KG augmentation in action! See `PHASE3_EXECUTION_GUIDE.md`

