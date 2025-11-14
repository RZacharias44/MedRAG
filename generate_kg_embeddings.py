#!/usr/bin/env python3
"""
Generate Knowledge Graph symptom embeddings only.

This script will:
1. Load the Knowledge Graph from Excel
2. Extract unique symptoms
3. Generate embeddings using OpenAI's text-embedding-3-large
4. Save them to ./Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy

Cost: ~$0.02
Time: ~5 minutes
"""

import os
import sys

print("=" * 70)
print("Knowledge Graph Symptom Embedding Generation")
print("=" * 70)

# Check API key first
print("\n[1/3] Checking API configuration...")
try:
    from authentication import api_key
    if not api_key or api_key == "" or api_key == "your-openai-api-key-here":
        print("✗ OpenAI API key not configured!")
        print("  Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    print("✓ API key loaded")
except Exception as e:
    print(f"✗ Error loading API key: {e}")
    sys.exit(1)

# Import and trigger KG embedding generation
print("\n[2/3] Loading Knowledge Graph and generating embeddings...")
print("This will:")
print("  • Load the DDXPlus knowledge graph")
print("  • Extract unique symptoms")
print("  • Generate embeddings for each symptom")
print("  • Save to: ./Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy")
print("\nStarting generation...\n")

try:
    # This import will trigger the embedding generation automatically
    # See KG_Retrieve.py line 84: symptom_embeddings = get_symptom_embeddings(...)
    import KG_Retrieve
    
    print("\n[3/3] Verifying embeddings were saved...")
    embedding_path = './Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy'
    
    if os.path.exists(embedding_path):
        import numpy as np
        embeddings = np.load(embedding_path)
        print(f"✓ Embeddings saved successfully!")
        print(f"  • Path: {embedding_path}")
        print(f"  • Shape: {embeddings.shape}")
        print(f"  • Number of symptoms: {embeddings.shape[0]}")
        print(f"  • Embedding dimension: {embeddings.shape[1]}")
    else:
        print(f"✗ Embeddings file not found at: {embedding_path}")
        sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error during embedding generation: {e}")
    print("\nIf you see a quota error, add credits to your OpenAI account:")
    print("  https://platform.openai.com/account/billing/overview")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ KG EMBEDDINGS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nNext steps:")
print("  1. These embeddings will be automatically loaded during diagnosis")
print("  2. Ready to run the full experiment with: uv run python main_MedRAG.py")
print("  3. Or test on a small subset first")
print("=" * 70)

