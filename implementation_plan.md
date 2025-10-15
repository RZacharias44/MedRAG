Action Plan: Adapting MedRAG to Run with the DDXPlus Dataset (v2)

This document outlines the necessary steps to modify the MedRAG codebase, shifting its focus from the unavailable private CPDD (Chronic Pain Diagnostic Dataset) to the public DDXPlus dataset. This version incorporates the exact preprocessing and sampling methodology described in the original paper to ensure a faithful replication.
Phase 1: Data Preparation & Sampling

This phase focuses on downloading the raw DDXPlus data and then creating the specific, balanced subset of patient records that the authors used for their experiments.
Step 1: Download the DDXPlus Dataset

    Action: Go to the DDXPlus Figshare page linked in the project's README.md.

    Download: Download the complete dataset. You will likely get a large CSV or a collection of files. For this plan, we will assume you have a primary CSV file named ddxplus_data.csv.

Step 2: Preprocess and Sample the Dataset

This is the most important new step. The following script will perform the sampling described in the paper: 240 samples per pathology for the training set and 30 for the test set.

    Action: Create a new Python script named preprocess_ddxplus.py in your project's root directory. Copy the code below into that file.

    Before you run:

        Make sure you have pandas installed (pip install pandas).

        Verify the column names in the script ('PATHOLOGY', 'PATIENT_ID', etc.) match the actual column names in your downloaded ddxplus_data.csv. You will likely need to adjust these.

# In preprocess_ddxplus.py

import pandas as pd
import os
import json

# --- CONFIGURATION: Adjust these variables ---
RAW_DATA_FILE = './dataset/ddxplus_data.csv' # The path to the large dataset you downloaded
PATHOLOGY_COLUMN = 'PATHOLOGY'                # The name of the column with the diagnosis
PATIENT_ID_COLUMN = 'PATIENT_ID'              # The name of the column with the patient ID

TRAIN_SAMPLES_PER_PATHOLOGY = 240
TEST_SAMPLES_PER_PATHOLOGY = 30
RANDOM_SEED = 42

# --- OUTPUT DIRECTORIES ---
TRAIN_DIR = './dataset/DDXPlus/train'
TEST_DIR = './dataset/DDXPlus/test'
GROUND_TRUTH_FILE = './dataset/DDXPlus_ground_truth.csv'

# Create output directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# --- SCRIPT LOGIC ---
print("Loading raw DDXPlus data...")
df = pd.read_csv(RAW_DATA_FILE)
print("Data loaded successfully.")

# Group data by pathology
grouped = df.groupby(PATHOLOGY_COLUMN)

all_train_dfs = []
all_test_dfs = []

print(f"Sampling data for {len(grouped)} pathologies...")

for pathology_name, group in grouped:
    # Ensure we have enough data to sample
    if len(group) < (TRAIN_SAMPLES_PER_PATHOLOGY + TEST_SAMPLES_PER_PATHOLOGY):
        print(f"Warning: Pathology '{pathology_name}' has only {len(group)} samples. Skipping.")
        continue
        
    # Sample the data according to the paper's methodology
    # First, sample the combined amount
    sampled_group = group.sample(n=(TRAIN_SAMPLES_PER_PATHOLOGY + TEST_SAMPLES_PER_PATHOLOGY), random_state=RANDOM_SEED)
    
    # Then, split into train and test sets
    test_set = sampled_group.sample(n=TEST_SAMPLES_PER_PATHOLOGY, random_state=RANDOM_SEED)
    train_set = sampled_group.drop(test_set.index)

    all_train_dfs.append(train_set)
    all_test_dfs.append(test_set)

# Concatenate all sampled data back into single dataframes
final_train_df = pd.concat(all_train_dfs)
final_test_df = pd.concat(all_test_dfs)

print(f"Total training samples: {len(final_train_df)}")
print(f"Total test samples: {len(final_test_df)}")


# --- FILE GENERATION ---

# 1. Save individual JSON files for the training set
print("Generating training JSON files...")
for index, row in final_train_df.iterrows():
    patient_id = row[PATIENT_ID_COLUMN]
    file_path = os.path.join(TRAIN_DIR, f"participant_{patient_id}.json")
    # Convert row to a dictionary and save as JSON
    row.to_json(file_path, orient='index')

# 2. Save individual JSON files for the test set
print("Generating test JSON files...")
for index, row in final_test_df.iterrows():
    patient_id = row[PATIENT_ID_COLUMN]
    file_path = os.path.join(TEST_DIR, f"participant_{patient_id}.json")
    row.to_json(file_path, orient='index')

# 3. Create the ground truth CSV from the test set
print("Generating ground truth CSV file...")
ground_truth_data = []
for index, row in final_test_df.iterrows():
    ground_truth_data.append({
        'PATIENT_ID': row[PATIENT_ID_COLUMN],
        'Filtered_Diagnoses': f"['{row[PATHOLOGY_COLUMN]}']"
    })

gt_df = pd.DataFrame(ground_truth_data)
gt_df.to_csv(GROUND_TRUTH_FILE, index=False)

print("\nPreprocessing complete!")
print(f"Train/Test files created in ./dataset/DDXPlus/")
print(f"Ground truth file created at {GROUND_TRUTH_FILE}")


    Run the Script: Execute python preprocess_ddxplus.py. This will create all the necessary files in the correct locations and formats.

Step 3: Validate Data Consistency

This step is now even more important to ensure your sampled data aligns with the knowledge graph.

    Action: Run the validation script from the previous version of the plan. It will check your newly created DDXPlus_ground_truth.csv against the KG. Fix any mismatches if they occur.

Phase 2: Code Configuration

This phase remains the same, as our preprocessing script has already placed the files where they need to go. You just need to tell the MedRAG code where to find them.
Step 1: Update File Paths in authentication.py

    Action: Open authentication.py and modify the path variables to match the output of your preprocessing script.

    # In authentication.py
    ob_path='./dataset/DDXPlus/train'
    test_folder_path="./dataset/DDXPlus/test"
    ground_truth_file_path='./dataset/DDXPlus_ground_truth.csv'
    augmented_features_path='./dataset/knowledge graph of DDXPlus.xlsx - Sheet1.csv'

Step 2: Update Paths in KG_Retrieve.py

    Action: Open KG_Retrieve.py and ensure its paths are correct.

    # In KG_Retrieve.py
    KG_file_path = './dataset/knowledge graph of DDXPlus.xlsx - Sheet1.csv'
    file_path = './dataset/DDXPlus_ground_truth.csv'

Step 3: Update the System Prompt in main_MedRAG.py

    Action: As before, open main_MedRAG.py, find get_system_prompt_for_RAGKG(), and replace the hardcoded disease list with the correct list from the DDXPlus KG.

Phase 3: Execution

With the data correctly sampled and the code configured, you are ready to run the replication.

    Install Dependencies: pip install -r requirements.txt

    Set API Keys: Add your OpenAI API key in authentication.py.

    Run the Main Script: python main.py

By following this updated plan, you are not just running the code on the dataset—you are precisely recreating the experimental conditions described in the paper, which is the gold standard for a replication study.