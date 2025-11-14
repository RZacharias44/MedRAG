#!/usr/bin/env python3
"""
Test MedRAG pipeline on a small subset of DDXPlus test patients.

This script runs the complete pipeline (retrieval + KG augmentation + LLM diagnosis)
on just 10 patients to verify everything works before the full experiment.

Cost: ~$0.10 - $0.20 for 10 patients
Time: ~5-10 minutes
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm
from main_MedRAG import (
    get_query_embedding, 
    Faiss, 
    extract_diagnosis, 
    documents, 
    document_embeddings,
    generate_diagnosis_report, 
    save_results_to_csv, 
    get_additional_info_from_level_2,
    KG_preprocess, 
    get_embeddings
)
from authentication import ob_path, test_folder_path, ground_truth_file_path, augmented_features_path

print("=" * 70)
print("MedRAG Small Subset Test - DDXPlus")
print("=" * 70)

# Test parameters
NUM_TEST_PATIENTS = 10  # Change this to test more patients
topk = 1  # Number of similar training cases to retrieve
top_n = 1  # Top N symptoms from KG
match_n = 5  # Number of KG matches
model = "gpt-4o-mini"  # LLM model to use (gpt-4o-mini is cheaper for testing)

print(f"\nTest Configuration:")
print(f"  • Number of test patients: {NUM_TEST_PATIENTS}")
print(f"  • LLM model: {model}")
print(f"  • Top-k retrieval: {topk}")
print(f"  • Top-n symptoms: {top_n}")
print(f"  • KG matches: {match_n}")
print(f"  • Test folder: {test_folder_path}")
print(f"  • Ground truth: {ground_truth_file_path}")

# Verify embeddings are loaded
print(f"\n✓ Embeddings loaded:")
print(f"  • Training documents: {len(documents)}")
print(f"  • Document embeddings: {document_embeddings.shape}")

# Load ground truth
ground_truth = pd.read_csv(ground_truth_file_path, header=0)
print(f"  • Ground truth entries: {len(ground_truth)}")

# Get list of test files
test_files = sorted([f for f in os.listdir(test_folder_path) if f.endswith('.json')])
print(f"  • Test files available: {len(test_files)}")

if len(test_files) < NUM_TEST_PATIENTS:
    print(f"\n⚠️  Warning: Only {len(test_files)} test files found, adjusting test size")
    NUM_TEST_PATIENTS = len(test_files)

# Select first N test patients
test_subset = test_files[:NUM_TEST_PATIENTS]

print(f"\n{'=' * 70}")
print(f"Processing {NUM_TEST_PATIENTS} test patients...")
print(f"{'=' * 70}\n")

results = []

for idx, test_file in enumerate(tqdm(test_subset, desc="Testing patients")):
    file_path = os.path.join(test_folder_path, test_file)
    
    # Extract participant number from filename
    # Expected format: participant_N.json or similar
    participant_no = test_file.replace('.json', '').replace('participant_', '')
    
    print(f"\n{'─' * 70}")
    print(f"[{idx+1}/{NUM_TEST_PATIENTS}] Processing: {test_file}")
    print(f"Participant No: {participant_no}")
    
    try:
        # Load patient case
        with open(file_path, 'r') as file:
            new_patient_case = json.load(file)
        
        # Get participant number from JSON if available
        if 'Participant No.' in new_patient_case:
            participant_no = new_patient_case['Participant No.']
        elif 'PATIENT' in new_patient_case:
            participant_no = new_patient_case['PATIENT']
        
        print(f"  • Patient symptoms loaded")
        
        # Create query from patient case
        query = json.dumps(new_patient_case)
        
        # Step 1: Generate query embedding
        print(f"  • Generating query embedding...")
        query_embedding = get_query_embedding(query)
        
        # Step 2: Retrieve similar training cases
        print(f"  • Retrieving top-{topk} similar training cases...")
        indices = Faiss(document_embeddings, query_embedding, k=topk)
        retrieved_documents = [documents[i] for i in indices[0]]
        
        # Step 3: Extract info from retrieved documents
        print(f"  • Extracting information from retrieved cases...")
        final_retrieved_info = []
        for retrieved_document in retrieved_documents:
            with open(retrieved_document, 'r') as file:
                patient_case = json.load(file)
                final_retrieved_info.append(patient_case)
        
        # Step 4: Get ground truth
        true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == int(participant_no)]
        if true_diagnosis_row.empty:
            print(f"  ✗ Ground truth not found for participant {participant_no}")
            true_diagnosis = "Not found"
            ori_truth = "Not found"
        else:
            true_diagnosis = true_diagnosis_row['Processed Diagnosis'].values[0]
            ori_truth = true_diagnosis_row['Diagnoses (related to pain)'].values[0]
            print(f"  • Ground truth: {true_diagnosis}")
        
        # Step 5: Generate diagnosis with KG augmentation
        print(f"  • Generating diagnosis with KG augmentation...")
        generated_report = generate_diagnosis_report(
            augmented_features_path,
            query, 
            final_retrieved_info, 
            idx,
            top_n=top_n,
            match_n=match_n,
            model=model
        )
        
        # Extract diagnosis from report
        generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report)
        
        if not generated_diagnosis:
            print(f"  ✗ No diagnosis extracted from report")
            generated_diagnosis_str = ""
        else:
            generated_diagnosis_str = generated_diagnosis[0]
            print(f"  ✓ Generated diagnosis: {generated_diagnosis_str}")
        
        # Store results
        results.append([
            participant_no, 
            generated_diagnosis_str, 
            true_diagnosis, 
            ori_truth,
            generated_report
        ])
        
        print(f"  ✓ Success!")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append([participant_no, '', 'Error', 'Error', str(e)])

# Save results
output_file = f"./test_results_subset_{NUM_TEST_PATIENTS}_patients.csv"
df = pd.DataFrame(results, columns=[
    'Participant No.', 
    'Generated Diagnosis', 
    'True Diagnosis', 
    'Ori Truth',
    'Generated Report'
])
df.to_csv(output_file, index=False)

print(f"\n{'=' * 70}")
print("✓ TEST COMPLETE!")
print(f"{'=' * 70}")
print(f"\nResults saved to: {output_file}")
print(f"\nSummary:")
print(f"  • Total patients tested: {len(results)}")
print(f"  • Successful diagnoses: {sum(1 for r in results if r[1] != '')}")
print(f"  • Failed: {sum(1 for r in results if r[1] == '')}")

# Calculate simple accuracy (exact match)
correct = sum(1 for r in results if r[1] and r[2] and str(r[1]).strip().lower() == str(r[2]).strip().lower())
print(f"  • Exact matches: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")

print(f"\nYou can now:")
print(f"  1. Review results in: {output_file}")
print(f"  2. If everything looks good, run the full experiment on all 1,470 patients")
print(f"  3. Modify test parameters (NUM_TEST_PATIENTS, topk, top_n, match_n)")
print(f"{'=' * 70}")

