"""
Data Validation Script for DDXPlus Preprocessing
=================================================

This script validates that the preprocessing was successful and matches
the paper's methodology. It checks:

1. Sample counts (13,230 total: 11,760 train + 1,470 test)
2. Balanced distribution (30 test + 240 train per pathology)
3. All pathologies exist in the knowledge graph
4. JSON file structure is correct
5. Ground truth consistency

Usage:
    uv run validate_preprocessing.py
"""

import os
import json
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

# Configuration
TRAIN_DIR = './dataset/DDXPlus/train'
TEST_DIR = './dataset/DDXPlus/test'
GROUND_TRUTH_FILE = './dataset/DDXPlus_ground_truth.csv'
CONDITIONS_FILE = './dataset/release_conditions.json'
KG_EXCEL_FILE = './dataset/knowledge graph of DDXPlus.xlsx'

# Expected values from paper
EXPECTED_TRAIN_PER_PATHOLOGY = 240
EXPECTED_TEST_PER_PATHOLOGY = 30
EXPECTED_TOTAL_PATHOLOGIES = 49
EXPECTED_TOTAL_SAMPLES = 13230


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"  {text}")


def validate_file_counts() -> Tuple[int, int, bool]:
    """Validate the number of JSON files in train and test directories"""
    print_header("Step 1: Validating File Counts")
    
    success = True
    
    # Count training files
    if not os.path.exists(TRAIN_DIR):
        print_error(f"Training directory not found: {TRAIN_DIR}")
        return 0, 0, False
    
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.json')]
    train_count = len(train_files)
    expected_train = EXPECTED_TOTAL_PATHOLOGIES * EXPECTED_TRAIN_PER_PATHOLOGY
    
    if train_count == expected_train:
        print_success(f"Training files: {train_count} (expected: {expected_train})")
    else:
        print_warning(f"Training files: {train_count} (expected: {expected_train})")
        success = False
    
    # Count test files
    if not os.path.exists(TEST_DIR):
        print_error(f"Test directory not found: {TEST_DIR}")
        return train_count, 0, False
    
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.json')]
    test_count = len(test_files)
    expected_test = EXPECTED_TOTAL_PATHOLOGIES * EXPECTED_TEST_PER_PATHOLOGY
    
    if test_count == expected_test:
        print_success(f"Test files: {test_count} (expected: {expected_test})")
    else:
        print_warning(f"Test files: {test_count} (expected: {expected_test})")
        success = False
    
    # Total count
    total = train_count + test_count
    if total == EXPECTED_TOTAL_SAMPLES:
        print_success(f"Total samples: {total} (expected: {EXPECTED_TOTAL_SAMPLES})")
    else:
        print_warning(f"Total samples: {total} (expected: {EXPECTED_TOTAL_SAMPLES})")
        success = False
    
    return train_count, test_count, success


def validate_ground_truth() -> Tuple[pd.DataFrame, bool]:
    """Validate the ground truth CSV file"""
    print_header("Step 2: Validating Ground Truth File")
    
    success = True
    
    if not os.path.exists(GROUND_TRUTH_FILE):
        print_error(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
        return None, False
    
    try:
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
        print_success(f"Ground truth file loaded: {len(gt_df)} entries")
    except Exception as e:
        print_error(f"Failed to load ground truth: {e}")
        return None, False
    
    # Check required columns
    required_columns = ['Participant No.', 'Processed Diagnosis', 'Diagnoses (related to pain)']
    missing_cols = [col for col in required_columns if col not in gt_df.columns]
    
    if missing_cols:
        print_error(f"Missing columns: {missing_cols}")
        success = False
    else:
        print_success(f"All required columns present: {required_columns}")
    
    # Check for duplicates
    if gt_df['Participant No.'].duplicated().any():
        print_error("Duplicate participant numbers found!")
        success = False
    else:
        print_success("No duplicate participant numbers")
    
    # Check pathology distribution
    pathology_counts = gt_df['Processed Diagnosis'].value_counts()
    unique_pathologies = len(pathology_counts)
    
    print_info(f"Unique pathologies: {unique_pathologies}")
    
    if unique_pathologies == EXPECTED_TOTAL_PATHOLOGIES:
        print_success(f"Pathology count matches expected: {EXPECTED_TOTAL_PATHOLOGIES}")
    else:
        print_warning(f"Pathology count: {unique_pathologies} (expected: {EXPECTED_TOTAL_PATHOLOGIES})")
        success = False
    
    # Check if all pathologies have exactly 30 samples (test set)
    expected_test = EXPECTED_TEST_PER_PATHOLOGY
    unbalanced = pathology_counts[pathology_counts != expected_test]
    
    if len(unbalanced) == 0:
        print_success(f"All pathologies have exactly {expected_test} test samples (balanced)")
    else:
        print_warning(f"Unbalanced pathologies found:")
        for pathology, count in unbalanced.items():
            print_info(f"  - {pathology}: {count} samples (expected {expected_test})")
        success = False
    
    return gt_df, success


def validate_json_structure(sample_size: int = 10) -> bool:
    """Validate JSON file structure for a sample of files"""
    print_header(f"Step 3: Validating JSON Structure (sample size: {sample_size})")
    
    success = True
    required_fields = [
        'Participant No.',
        'Processed Diagnosis',
        'Age',
        'Sex',
        'Evidences',
        'Initial Evidence',
        'Differential Diagnosis'
    ]
    
    # Sample from both train and test
    train_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) 
                   if f.endswith('.json')][:sample_size]
    test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) 
                  if f.endswith('.json')][:sample_size]
    
    all_files = train_files + test_files
    
    for i, file_path in enumerate(all_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            missing = [field for field in required_fields if field not in data]
            if missing:
                print_error(f"File {os.path.basename(file_path)} missing fields: {missing}")
                success = False
            
            # Check data types
            if not isinstance(data.get('Participant No.'), int):
                print_warning(f"File {os.path.basename(file_path)}: Participant No. is not an integer")
                success = False
            
            if data.get('Age') is not None and not isinstance(data.get('Age'), int):
                print_warning(f"File {os.path.basename(file_path)}: Age is not an integer")
                success = False
            
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in {os.path.basename(file_path)}: {e}")
            success = False
        except Exception as e:
            print_error(f"Error reading {os.path.basename(file_path)}: {e}")
            success = False
    
    if success:
        print_success(f"All {len(all_files)} sampled JSON files have correct structure")
    
    return success


def validate_pathology_distribution() -> Tuple[Dict, bool]:
    """Validate pathology distribution across train and test sets"""
    print_header("Step 4: Validating Pathology Distribution")
    
    success = True
    
    # Load all diagnoses from train set
    train_diagnoses = []
    for filename in os.listdir(TRAIN_DIR):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(TRAIN_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                train_diagnoses.append(data['Processed Diagnosis'])
        except Exception as e:
            print_error(f"Error reading {filename}: {e}")
            success = False
    
    # Load all diagnoses from test set
    test_diagnoses = []
    for filename in os.listdir(TEST_DIR):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(TEST_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_diagnoses.append(data['Processed Diagnosis'])
        except Exception as e:
            print_error(f"Error reading {filename}: {e}")
            success = False
    
    # Count occurrences
    train_counts = Counter(train_diagnoses)
    test_counts = Counter(test_diagnoses)
    
    print_info(f"Training set: {len(train_counts)} unique pathologies")
    print_info(f"Test set: {len(test_counts)} unique pathologies")
    
    # Check if all pathologies appear in both sets
    train_only = set(train_counts.keys()) - set(test_counts.keys())
    test_only = set(test_counts.keys()) - set(train_counts.keys())
    
    if train_only:
        print_warning(f"Pathologies only in training set: {train_only}")
        success = False
    
    if test_only:
        print_warning(f"Pathologies only in test set: {test_only}")
        success = False
    
    if not train_only and not test_only:
        print_success("All pathologies appear in both train and test sets")
    
    # Check balance
    print_info("\nChecking balance:")
    unbalanced_train = {k: v for k, v in train_counts.items() 
                        if v != EXPECTED_TRAIN_PER_PATHOLOGY}
    unbalanced_test = {k: v for k, v in test_counts.items() 
                       if v != EXPECTED_TEST_PER_PATHOLOGY}
    
    if unbalanced_train:
        print_warning(f"Unbalanced training pathologies ({len(unbalanced_train)}):")
        for pathology, count in list(unbalanced_train.items())[:5]:
            print_info(f"  - {pathology}: {count} (expected {EXPECTED_TRAIN_PER_PATHOLOGY})")
        if len(unbalanced_train) > 5:
            print_info(f"  ... and {len(unbalanced_train) - 5} more")
        success = False
    else:
        print_success(f"All training pathologies have exactly {EXPECTED_TRAIN_PER_PATHOLOGY} samples")
    
    if unbalanced_test:
        print_warning(f"Unbalanced test pathologies ({len(unbalanced_test)}):")
        for pathology, count in list(unbalanced_test.items())[:5]:
            print_info(f"  - {pathology}: {count} (expected {EXPECTED_TEST_PER_PATHOLOGY})")
        if len(unbalanced_test) > 5:
            print_info(f"  ... and {len(unbalanced_test) - 5} more")
        success = False
    else:
        print_success(f"All test pathologies have exactly {EXPECTED_TEST_PER_PATHOLOGY} samples")
    
    distribution = {
        'train': train_counts,
        'test': test_counts
    }
    
    return distribution, success


def validate_against_knowledge_graph(gt_df: pd.DataFrame) -> bool:
    """Validate that all pathologies exist in the knowledge graph"""
    print_header("Step 5: Validating Against Knowledge Graph")
    
    success = True
    
    # Try to load conditions from JSON (most reliable)
    if os.path.exists(CONDITIONS_FILE):
        try:
            with open(CONDITIONS_FILE, 'r', encoding='utf-8') as f:
                conditions = json.load(f)
            
            kg_pathologies = set(conditions.keys())
            print_success(f"Knowledge graph loaded: {len(kg_pathologies)} conditions from JSON")
            
            # Check if all ground truth pathologies exist in KG
            gt_pathologies = set(gt_df['Processed Diagnosis'].unique())
            
            missing_in_kg = gt_pathologies - kg_pathologies
            extra_in_gt = kg_pathologies - gt_pathologies
            
            if missing_in_kg:
                print_error(f"Pathologies in dataset but NOT in knowledge graph ({len(missing_in_kg)}):")
                for pathology in list(missing_in_kg)[:10]:
                    print_info(f"  - {pathology}")
                if len(missing_in_kg) > 10:
                    print_info(f"  ... and {len(missing_in_kg) - 10} more")
                success = False
            else:
                print_success("All dataset pathologies exist in knowledge graph")
            
            print_info(f"\nKnowledge graph has {len(extra_in_gt)} additional pathologies not in dataset")
            
        except Exception as e:
            print_error(f"Failed to load conditions JSON: {e}")
            success = False
    else:
        print_warning(f"Conditions file not found: {CONDITIONS_FILE}")
        print_info("Skipping knowledge graph validation")
    
    return success


def print_summary(results: Dict[str, bool]):
    """Print overall validation summary"""
    print_header("Validation Summary")
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    for check_name, passed in results.items():
        if passed:
            print_success(check_name)
        else:
            print_error(check_name)
    
    print(f"\n{Colors.BOLD}Overall: {passed_checks}/{total_checks} checks passed{Colors.END}")
    
    if passed_checks == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All validations passed! Data preprocessing is correct.{Colors.END}")
        return True
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some validations failed. Please review the errors above.{Colors.END}")
        return False


def main():
    """Run all validation checks"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print("DDXPlus Preprocessing Validation")
    print("="*70)
    print(f"{Colors.END}")
    
    results = {}
    
    # Step 1: File counts
    train_count, test_count, success1 = validate_file_counts()
    results["File counts"] = success1
    
    # Step 2: Ground truth
    gt_df, success2 = validate_ground_truth()
    results["Ground truth structure"] = success2
    
    # Step 3: JSON structure
    success3 = validate_json_structure(sample_size=20)
    results["JSON structure"] = success3
    
    # Step 4: Pathology distribution
    distribution, success4 = validate_pathology_distribution()
    results["Pathology distribution"] = success4
    
    # Step 5: Knowledge graph consistency
    if gt_df is not None:
        success5 = validate_against_knowledge_graph(gt_df)
        results["Knowledge graph consistency"] = success5
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

