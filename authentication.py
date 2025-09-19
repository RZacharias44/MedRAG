ob_path='./dataset/DDXPlus/train'
test_folder_path="./dataset/DDXPlus/test"
ground_truth_file_path='./dataset/DDXPlus_ground_truth.csv'
augmented_features_path='./dataset/knowledge graph of DDXPlus.xlsx'

import os
from dotenv import load_dotenv

# Load environment variables from a local .env file if present
load_dotenv()

# API tokens are read from environment variables to avoid committing secrets
# Set these in your shell or a local .env file
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_TOKEN")
