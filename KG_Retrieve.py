
import openai
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

from authentication import api_key
client = openai.OpenAI(api_key=api_key)

from authentication import augmented_features_path, ground_truth_file_path

KG_file_path = augmented_features_path  # './dataset/knowledge graph of DDXPlus.xlsx'
file_path = ground_truth_file_path  # './dataset/DDXPlus_ground_truth.csv'
embedding_save_path = './Embeddings_saved/DDXPlus_KG_embeddings'



def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)


kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])

knowledge_graph = {}
for index, row in kg_data.iterrows():
    subject = row['subject']
    relation = row['relation']
    obj = row['object']

    if subject not in knowledge_graph:
        knowledge_graph[subject] = []
    knowledge_graph[subject].append((relation, obj))

    if obj not in knowledge_graph:
        knowledge_graph[obj] = []
    knowledge_graph[obj].append((relation, subject))

kg_data['object_preprocessed'] = kg_data.apply(
    lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
    axis=1
)
symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()


def get_symptom_embeddings(symptom_nodes, save_path):
    embeddings_path = os.path.join(save_path, 'KG_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("load existing embeddings...")
        return np.load(embeddings_path)
    else:
        print("generate new embeddings...")
        symptom_embeddings = []
        for symptom in tqdm(symptom_nodes):
            response = client.embeddings.create(
                input=symptom,
                model="text-embedding-3-large"
            )
            symptom_embeddings.append(response.data[0].embedding)
        np.save(embeddings_path, symptom_embeddings)

        return np.array(symptom_embeddings)


symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)


def find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings, n):
    if pd.isna(query) or not query:
        return []
    query_preprocessed = preprocess_text(query)
    response = client.embeddings.create(
        input=query_preprocessed,
        model="text-embedding-3-large"
    )
    query_embedding = response.data[0].embedding
    if not query_embedding:
        return []

    if len(symptom_embeddings) > len(symptom_nodes):
        symptom_embeddings = symptom_embeddings[:len(symptom_nodes)]

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    top_n_symptoms = []
    unique_symptoms = set()
    top_n_indices = similarities.argsort()[::-1]

    for i in top_n_indices:
        if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
            top_n_symptoms.append(symptom_nodes[i])
            unique_symptoms.add(symptom_nodes[i])
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms


def compute_shortest_path_length(node1, node2, G):
    try:
        return nx.shortest_path_length(G, source=node1, target=node2)
    except nx.NetworkXNoPath:
        return float('inf')

categories = [
    "thoracoabdominal_pain_syndromes",
    "neuropathic_pain_syndromes",
    "craniofacial_pain_syndromes",
    "cervical_spine_pain_syndromes",
    "limb_and_joint_pain_syndromes",
    "back_pain_syndromes",
    "lumbar_degenerative_and_stenosis_and_radicular_and_sciatic_syndromes",
    "generalized_pain_syndromes",

]
G = nx.Graph()
for node, edges in knowledge_graph.items():
    for relation, neighbor in edges:
        G.add_edge(node, neighbor, relation=relation)


def get_diagnoses_for_symptom(symptom):

    diagnoses = []
    if symptom in G:
        for neighbor in G.neighbors(symptom):
            edge_data = G.get_edge_data(neighbor, symptom)
            if edge_data and 'relation' in edge_data and edge_data['relation'] != 'is_a':
                diagnoses.append(neighbor)
    return diagnoses


def find_closest_category(top_symptoms, categories,top_n):
    if isinstance(top_symptoms, pd.Series) and top_symptoms.empty:
        print("Warning: top_symptoms is empty.")
        return None
    category_votes = {category: 0 for category in categories}
    for symptom in top_symptoms:
        top_symptoms = list(set(top_symptoms))

        # print('symptom: ',symptom)
        if symptom not in G:
            print(f"Symptom node not found in graph: {symptom}")
            continue

        diagnosis_nodes = get_diagnoses_for_symptom(symptom)
        for diagnosis in diagnosis_nodes:

            individual_diagnoses = diagnosis.split(',')

            for single_diagnosis in individual_diagnoses:
                single_diagnosis = single_diagnosis.strip().replace(' ', '_').lower()  # 去掉前后空格
                if single_diagnosis not in G:
                    print(f"Diagnosis node not found in graph: {single_diagnosis}")
                    continue

                min_distance = float('inf')
                closest_category = None

                for category in categories:
                    if category not in G:
                        print(f"Category node not found in graph: {category}")
                        continue

                    try:
                        distance = nx.shortest_path_length(G, source=single_diagnosis, target=category)
                    except nx.NetworkXNoPath:
                        distance = float('inf')

                    if distance < min_distance:
                        min_distance = distance
                        closest_category = category

                if closest_category:
                    category_votes[closest_category] += 1
    print("Category votes:", category_votes)

    sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
    top_n_categories = [sorted_categories[i][0] for i in range(top_n)]
    return top_n_categories


def get_keyinfo_for_category(category, knowledge_graph):
    keyinfo_values = []
    for node, edges in knowledge_graph.items():
        if node == category:
            for relation, neighbor in edges:
                if relation == "is_a" and neighbor in knowledge_graph:
                    for rel, obj in knowledge_graph[neighbor]:
                        if rel == "has_keyinfo":
                            keyinfo_values.append(obj)
    return keyinfo_values



def get_subjects_for_objects(objects, knowledge_graph):
    subjects = []
    processed_objects = [obj.replace(' ', '_') for obj in objects]
    for obj in processed_objects:
        for index, row in knowledge_graph.iterrows():
            if row['object'] == obj:
                subjects.append(row['subject'])
    return subjects


def find_level3_for_symptoms(top_symptoms, knowledge_graph):
    level3_connections = {}
    for symptom in top_symptoms:
        subjects = get_subjects_for_objects([symptom], knowledge_graph)
        for subject in subjects:
            if subject in level3_connections:
                level3_connections[subject] += 1
            else:
                level3_connections[subject] = 1
    return level3_connections


def print_symptom_and_disease(symptom_nodes):
    for symptom in symptom_nodes:
        subjects = get_subjects_for_objects([symptom], kg_data)


def decode_ddxplus_symptom(symptom_code):
    """
    Convert DDXPlus symptom codes to readable text for KG matching.
    
    DDXPlus uses coded French abbreviations like:
    - douleurxx → pain
    - dyspn → dyspnea, shortness of breath
    - palpit → palpitations
    - douleurxx_endroitducorps_@_X → pain location X
    """
    # Basic symptom code mappings (French to English medical terms)
    symptom_map = {
        'douleurxx': 'pain',
        'dyspn': 'dyspnea shortness of breath',
        'palpit': 'palpitations',
        'fievre': 'fever',
        'toux': 'cough',
        'nausee': 'nausea',
        'vomissement': 'vomiting',
        'diarrhee': 'diarrhea',
        'fatigue': 'fatigue',
        'cephalee': 'headache',
        'vertiges': 'dizziness',
        'convulsions': 'seizures convulsions',
        'hemoptysie': 'hemoptysis coughing blood',
        'douleur_gorge': 'sore throat',
        'rhinorrhee': 'runny nose rhinorrhea',
        'otalgie': 'ear pain otalgia',
        'prurit': 'itching pruritus',
        'rash': 'rash skin eruption',
        'oedeme': 'edema swelling',
        'sueurs': 'sweating perspiration',
        'frissons': 'chills shivering',
        'asthenie': 'weakness asthenia',
        'anorexie': 'loss of appetite anorexia',
        'douleur_abdo': 'abdominal pain',
        'douleur_thorax': 'chest pain',
        'thorax': 'chest thorax',
        'abdomen': 'abdomen abdominal',
        'tete': 'head',
        'dos': 'back',
        'jambe': 'leg',
        'bras': 'arm',
    }
    
    # Clean the code
    code = symptom_code.lower().strip()
    
    # Handle location-specific codes (e.g., douleurxx_endroitducorps_@_bas_du_thorax)
    if '_endroitducorps_@_' in code:
        # Extract location
        parts = code.split('_endroitducorps_@_')
        symptom_base = parts[0] if parts else 'pain'
        location = parts[1] if len(parts) > 1 else ''
        location_clean = location.replace('_', ' ')
        return f"{symptom_map.get(symptom_base, symptom_base)} in {location_clean}"
    
    # Handle characteristic codes (e.g., douleurxx_carac_@_vive)
    if '_carac_@_' in code:
        parts = code.split('_carac_@_')
        symptom_base = parts[0] if parts else 'pain'
        characteristic = parts[1] if len(parts) > 1 else ''
        characteristic_clean = characteristic.replace('_', ' ')
        return f"{characteristic_clean} {symptom_map.get(symptom_base, symptom_base)}"
    
    # Handle intensity codes (e.g., douleurxx_intens_@_8)
    if '_intens_@_' in code:
        parts = code.split('_intens_@_')
        symptom_base = parts[0] if parts else 'pain'
        intensity = parts[1] if len(parts) > 1 else ''
        return f"severe {symptom_map.get(symptom_base, symptom_base)} intensity {intensity}"
    
    # Handle irradiation codes (e.g., douleurxx_irrad_@_colonne_dorsale)
    if '_irrad_@_' in code:
        parts = code.split('_irrad_@_')
        symptom_base = parts[0] if parts else 'pain'
        irrad_location = parts[1] if len(parts) > 1 else ''
        irrad_clean = irrad_location.replace('_', ' ')
        return f"{symptom_map.get(symptom_base, symptom_base)} radiating to {irrad_clean}"
    
    # Handle other @ codes
    if '_@_' in code:
        code = code.split('_@_')[0]
    
    # Remove underscores and try direct mapping
    code_clean = code.replace('_', ' ')
    
    # Try to find in symptom map
    for key, value in symptom_map.items():
        if key in code:
            return value
    
    # Return cleaned code if no mapping found
    return code_clean


def main_get_category_and_level3(n, participant_no, top_n):
    """
    Extract patient symptoms from DDXPlus data and match to Knowledge Graph.
    
    For DDXPlus:
    - Reads 'Evidences' field from test JSON files (not ground truth CSV)
    - Decodes symptom codes to readable text
    - Matches against KG using embeddings
    - Returns categories for diagnosis augmentation
    """
    # Import test_folder_path from authentication
    from authentication import test_folder_path
    
    # Read from test JSON file instead of ground truth CSV
    test_file_path = os.path.join(test_folder_path, f'participant_{participant_no}.json')
    
    if not os.path.exists(test_file_path):
        print(f"Test file not found for participant {participant_no}: {test_file_path}")
        return ['thoracoabdominal_pain_syndromes']  # Default category
    
    try:
        import json
        with open(test_file_path, 'r') as f:
            patient_data = json.load(f)
    except Exception as e:
        print(f"Error loading test file for participant {participant_no}: {e}")
        return ['thoracoabdominal_pain_syndromes']  # Default category

    # Extract DDXPlus Evidences field from JSON
    evidences_str = patient_data.get("Evidences", '')
    initial_evidence = patient_data.get("Initial Evidence", '')
    
    # Parse the Evidences string (it's a string representation of a Python list)
    symptom_codes = []
    if evidences_str and not pd.isna(evidences_str):
        try:
            # Use ast.literal_eval to safely parse the string list
            import ast
            symptom_codes = ast.literal_eval(evidences_str)
        except:
            # Fallback: manual parsing
            evidences_str = evidences_str.strip("[]'\"")
            symptom_codes = [s.strip().strip("'\"") for s in evidences_str.split(',') if s.strip()]
    
    # Add initial evidence if present
    if initial_evidence and not pd.isna(initial_evidence):
        symptom_codes.append(initial_evidence)
    
    print(f'Found {len(symptom_codes)} symptom codes for patient {participant_no}')
    
    # Decode symptom codes to readable text
    decoded_symptoms = []
    for code in symptom_codes[:20]:  # Limit to first 20 symptoms to avoid too much processing
        decoded = decode_ddxplus_symptom(code)
        if decoded:
            decoded_symptoms.append(decoded)
    
    print(f'Decoded symptoms: {decoded_symptoms[:5]}...')  # Show first 5
    
    # Combine all decoded symptoms into a single text for KG matching
    combined_symptoms_text = ' '.join(decoded_symptoms)
    
    if not combined_symptoms_text:
        print('No symptoms found after decoding')
        return ['thoracoabdominal_pain_syndromes']  # Default category
    
    # Helper function to process symptom field
    def process_symptom_field(field_value, symptom_nodes, symptom_embeddings, n):
        if pd.isna(field_value) or field_value == '':
            return []
        return find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n)
    
    # Match decoded symptoms against KG using embeddings
    matched_symptom_nodes = process_symptom_field(
        combined_symptoms_text,
        symptom_nodes,
        symptom_embeddings,
        n * 3  # Get more matches since we're combining symptoms
    )
    
    print(f'Matched {len(matched_symptom_nodes)} KG symptom nodes')
    
    if not matched_symptom_nodes:
        print('No KG symptom matches found')
        return ['thoracoabdominal_pain_syndromes']  # Default category
    
    # Get original symptom text from KG
    matched_symptoms_original = kg_data.loc[
        kg_data['object_preprocessed'].isin(matched_symptom_nodes),
        'object'
    ].drop_duplicates().tolist()
    
    print(f'Matched original symptoms: {matched_symptoms_original[:3]}...')  # Show first 3
    
    # For DDXPlus: Return matched symptoms directly instead of trying to find pain categories
    # The categories (thoracoabdominal_pain_syndromes, etc.) don't exist in DDXPlus KG
    # Instead, we return the actual matched symptom descriptions from the KG
    
    if not matched_symptoms_original:
        print('No matched symptoms to return')
        return []
    
    print(f'Returning {len(matched_symptoms_original)} matched symptoms as KG augmentation')
    
    # Return the matched symptom descriptions (these will be used to augment the diagnosis prompt)
    return matched_symptoms_original[:top_n * 3]  # Return top symptoms for augmentation
