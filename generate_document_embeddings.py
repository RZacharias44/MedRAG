#!/usr/bin/env python3
"""
Generate training document embeddings only.

This script will:
1. Load all 11,760 training patient JSON files
2. Generate embeddings using OpenAI's text-embedding-3-large
3. Save them to ./Embeddings_saved/DDXPlus_document_embeddings.npy

Cost: ~$0.74
Time: ~5-10 minutes
"""

import os
import sys
import numpy as np

print("=" * 70)
print("Training Document Embedding Generation")
print("=" * 70)

# Check API key first
print("\n[1/4] Checking API configuration...")
try:
    from authentication import api_key, ob_path
    if not api_key or api_key == "" or api_key == "your-openai-api-key-here":
        print("✗ OpenAI API key not configured!")
        print("  Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    print("✓ API key loaded")
except Exception as e:
    print(f"✗ Error loading API key: {e}")
    sys.exit(1)

# Check if embeddings already exist
document_embeddings_file_path = './Embeddings_saved/DDXPlus_document_embeddings.npy'

print("\n[2/4] Checking for existing embeddings...")
if os.path.exists(document_embeddings_file_path):
    embeddings = np.load(document_embeddings_file_path)
    print(f"✓ Embeddings already exist!")
    print(f"  • Path: {document_embeddings_file_path}")
    print(f"  • Shape: {embeddings.shape}")
    print(f"  • Number of documents: {embeddings.shape[0]}")
    print(f"  • Embedding dimension: {embeddings.shape[1]}")
    print("\n✓ NO GENERATION NEEDED - Embeddings already cached!")
    print("=" * 70)
    sys.exit(0)

# Load training documents
print("\n[3/4] Loading training documents...")
try:
    folder_path = ob_path
    documents = [os.path.join(folder_path, file_name) 
                 for file_name in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, file_name))]
    print(f"✓ Found {len(documents)} training documents")
    print(f"  • Source: {folder_path}")
except Exception as e:
    print(f"✗ Error loading documents: {e}")
    sys.exit(1)

# Generate embeddings
print("\n[4/4] Generating embeddings...")
print("This will:")
print("  • Read each patient JSON file")
print("  • Generate embedding for each patient")
print("  • Save to: ./Embeddings_saved/DDXPlus_document_embeddings.npy")
print(f"\nEstimated cost: ~$0.74")
print(f"Estimated time: ~5-10 minutes")
print("\nStarting generation...\n")

try:
    from main_MedRAG import get_embeddings
    
    # Generate embeddings with progress bar
    document_embeddings = get_embeddings(documents)
    
    # Save embeddings
    print(f"\n✓ Embeddings generated!")
    print(f"  Saving to: {document_embeddings_file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(document_embeddings_file_path), exist_ok=True)
    np.save(document_embeddings_file_path, document_embeddings)
    
    # Verify
    if os.path.exists(document_embeddings_file_path):
        saved_embeddings = np.load(document_embeddings_file_path)
        print(f"✓ Embeddings saved successfully!")
        print(f"  • Path: {document_embeddings_file_path}")
        print(f"  • Shape: {saved_embeddings.shape}")
        print(f"  • File size: {os.path.getsize(document_embeddings_file_path) / (1024*1024):.1f} MB")
    else:
        print(f"✗ Embeddings file not found after saving!")
        sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error during embedding generation: {e}")
    print("\nIf you see a quota error, add credits to your OpenAI account:")
    print("  https://platform.openai.com/account/billing/overview")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ DOCUMENT EMBEDDINGS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nEmbeddings Summary:")
print(f"  • KG embeddings: ./Embeddings_saved/DDXPlus_KG_embeddings/KG_embeddings.npy ✓")
print(f"  • Document embeddings: {document_embeddings_file_path} ✓")
print("\nNext steps:")
print("  1. Both embedding sets are now cached and ready to use")
print("  2. Run the full experiment with: uv run python main.py")
print("  3. Or modify main.py to test on a smaller subset first")
print("=" * 70)

