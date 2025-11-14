#!/usr/bin/env python3
"""
Quick test script to verify:
1. All dependencies are installed
2. OpenAI API key is configured correctly
3. Embedding generation works
4. File I/O works

This runs a minimal test before the full KG embedding generation.
"""

import sys
import os

print("=" * 60)
print("Testing MedRAG Dependencies & Embedding Generation")
print("=" * 60)

# Test 1: Import all required modules
print("\n[1/5] Testing imports...")
try:
    import openai
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    from sklearn.metrics.pairwise import cosine_similarity
    from dotenv import load_dotenv
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load environment and API key
print("\n[2/5] Testing API key configuration...")
try:
    from authentication import api_key
    if not api_key or api_key == "" or api_key == "your-openai-api-key-here":
        print("✗ OpenAI API key not configured!")
        print("  Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)
    print("✓ API key loaded")
except Exception as e:
    print(f"✗ API key error: {e}")
    sys.exit(1)

# Test 3: Download NLTK data if needed
print("\n[3/5] Testing NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    # Test tokenization
    test_tokens = word_tokenize("chest pain")
    print(f"✓ NLTK working (tokenized 'chest pain' → {test_tokens})")
except Exception as e:
    print(f"✗ NLTK error: {e}")
    sys.exit(1)

# Test 4: Test OpenAI API connection and embedding generation
print("\n[4/5] Testing OpenAI API connection & embedding generation...")
test_symptoms = [
    "chest pain",
    "headache",
    "fever"
]

try:
    print(f"  Generating embeddings for {len(test_symptoms)} test symptoms...")
    test_embeddings = []
    
    for i, symptom in enumerate(test_symptoms, 1):
        print(f"    [{i}/{len(test_symptoms)}] Embedding: '{symptom}'", end="")
        response = client.embeddings.create(
            input=symptom,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        test_embeddings.append(embedding)
        print(f" → {len(embedding)} dimensions ✓")
    
    test_embeddings = np.array(test_embeddings)
    print(f"✓ Generated {len(test_embeddings)} embeddings, shape: {test_embeddings.shape}")
    
except Exception as e:
    print(f"\n✗ OpenAI API error: {e}")
    print("\nPossible issues:")
    print("  - Invalid API key")
    print("  - No internet connection")
    print("  - Insufficient API credits")
    sys.exit(1)

# Test 5: Test file I/O
print("\n[5/5] Testing file I/O...")
try:
    test_dir = "./Embeddings_saved/test"
    os.makedirs(test_dir, exist_ok=True)
    
    test_file = os.path.join(test_dir, "test_embeddings.npy")
    np.save(test_file, test_embeddings)
    print(f"  Saved test embeddings to: {test_file}")
    
    loaded = np.load(test_file)
    print(f"  Loaded embeddings back, shape: {loaded.shape}")
    
    if np.array_equal(test_embeddings, loaded):
        print("✓ File I/O working correctly")
    else:
        print("✗ Data mismatch after loading")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ File I/O error: {e}")
    sys.exit(1)

# Test 6: Check Knowledge Graph file exists
print("\n[6/6] Checking Knowledge Graph file...")
try:
    from authentication import augmented_features_path
    if os.path.exists(augmented_features_path):
        kg_data = pd.read_excel(augmented_features_path, usecols=['subject', 'relation', 'object'])
        print(f"✓ Knowledge Graph loaded: {len(kg_data)} triples")
        
        # Count unique symptoms
        kg_data['object_preprocessed'] = kg_data.apply(
            lambda row: row['object'] if row['relation'] != 'is_a' else None,
            axis=1
        )
        symptom_count = kg_data['object_preprocessed'].dropna().nunique()
        print(f"  Total unique symptoms in KG: {symptom_count}")
        print(f"  → Full embedding generation will process ~{symptom_count} symptoms")
        
        # Estimate cost and time
        cost_per_1k = 0.00013  # text-embedding-3-large pricing
        estimated_cost = (symptom_count / 1000) * cost_per_1k
        estimated_time_min = symptom_count * 0.5 / 60  # ~0.5 sec per embedding
        
        print(f"\n  Estimated full run:")
        print(f"    • Time: ~{estimated_time_min:.1f} minutes")
        print(f"    • Cost: ~${estimated_cost:.3f}")
        
    else:
        print(f"✗ Knowledge Graph not found at: {augmented_features_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Knowledge Graph error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour environment is ready for full embedding generation.")
print("\nNext steps:")
print("  1. Clean up test files (optional):")
print("     rm -rf ./Embeddings_saved/test")
print("  2. Run full embedding generation:")
print("     uv run python KG_Retrieve.py")
print("     (or just import it in main_MedRAG.py)")
print("=" * 60)

