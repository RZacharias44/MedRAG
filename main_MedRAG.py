import openai
import faiss
import numpy as np
import os
import re
import json
import pandas as pd
from tqdm import tqdm
from huggingface_hub import InferenceClient
from KG_Retrieve import main_get_category_and_level3
from authentication import api_key,hf_token, ob_path

client = openai.OpenAI(api_key=api_key)

def get_embeddings(texts):
    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def get_query_embedding(query):
    return get_embeddings([query])[0]


# FAISS
def Faiss(document_embeddings, query_embedding, k):
    # index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    # index = faiss.IndexHNSWFlat(document_embeddings.shape[1])
    index.add(document_embeddings)
    _, indices = index.search(np.array([query_embedding]), k)
    print("index: ", indices)
    return indices

def extract_diagnosis(generated_text):
    diagnoses = re.findall(r'\*\*Diagnosis\*\*:\s(.*?)\n', generated_text)
    return diagnoses

def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()

def KG_preprocess(file_path):
    kg_data = pd.read_excel(file_path, usecols=['subject', 'relation', 'object'])
    kg_data['subject'] = kg_data['subject'].apply(remove_parentheses)
    kg_data['object'] = kg_data['object'].apply(remove_parentheses)

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
    return knowledge_graph


def extract_features_from_json(file_path):
    with open(file_path, 'r') as file:
        patient_case = json.load(file)

    pain_location = patient_case.get("Pain Presentation and Description Areas of pain as per physiotherapy input", "")
    pain_symptoms = patient_case.get(
        "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
        "")

    return pain_location, pain_symptoms

level_3_to_level_2 = {
    # Here are subcategories: diseases
    # Examples: 
    
    # Respiratory System
    "acute_copd_exacerbation_infection": "respiratory_system",

    # Cardiovascular System
    "atrial_fibrillation": "cardiovascular_system",

}


def get_additional_info_from_level_2(participant_no, kg_path, top_n, match_n):
    """
    Get KG augmentation for DDXPlus patients.
    
    For DDXPlus:
    - Extracts patient symptoms from Evidences field
    - Matches symptoms to Knowledge Graph using embeddings
    - Returns relevant symptom descriptions from KG to augment diagnosis
    """
    # Get matched KG symptoms for this patient
    matched_symptoms = main_get_category_and_level3(match_n, participant_no, top_n)
    
    if not matched_symptoms:
        print(f"No KG augmentation found for Participant No.: {participant_no}")
        return None
    
    # For DDXPlus: matched_symptoms are actual symptom descriptions from KG
    # These are already meaningful text that can augment the LLM prompt
    final_info = "Relevant symptoms and clinical presentations from Knowledge Graph: " + "; ".join(matched_symptoms)
    
    print(f"KG Augmentation ({len(matched_symptoms)} symptoms): {final_info[:200]}...")
    
    return final_info


def get_system_prompt_for_RAGKG():
    return '''
        You are a knowledgeable medical assistant with expertise in diagnostic medicine.
        Your tasks are:
        1. Analyse and refer to the retrieved similar patients' cases and knowledge graph which may be relevant to the diagnosis and assist with new patient cases.
        2. Output of "Diagnoses" must come from the following DDXPlus pathologies (use the EXACT French names):
           Anaphylaxie, Angine instable, Angine stable, Anémie, Asthme exacerbé ou bronchospasme, Attaque de panique, Bronchiectasies, Bronchiolite, Bronchite, Chagas, Coqueluche, Céphalée en grappe, Ebola, Embolie pulmonaire, Exacerbation aigue de MPOC et/ou surinfection associée, Fibrillation auriculaire/Flutter auriculaire, Fracture de côte spontanée, Hernie inguinale, IVRS ou virémie, Laryngite aigue, Laryngo-trachéo-bronchite (Croup), Laryngospasme, Lupus érythémateux disséminé (LED), Myasthénie grave, Myocardite, Néoplasie du pancréas, OAP/Surcharge pulmonaire, Oedème localisé ou généralisé sans atteinte pulmonaire associée, Otite moyenne aigue (OMA), Pharyngite virale, Pneumonie, Pneumothorax spontané, Possible NSTEMI / STEMI, Possible influenza ou syndrome virémique typique, Péricardite, RGO, Rhinite allergique, Rhinosinusite aigue, Rhinosinusite chronique, Réaction dystonique aïgue, Sarcoïdose, Scombroïde, Syndrome de Boerhaave, Syndrome de Guillain-Barré, TSVP, Tuberculose, VIH (Primo-infection), néoplasie pulmonaire, Épiglottite
        3. You are given differences of diagnoses of similar symptoms or clinical presentations. Read that information as a reference to your diagnostic if applicable.
        4. Consider the nuances between similar diagnoses using the knowledge graph information when diagnosing the new patient's condition.
        5. Ensure that the recommendations are evidence-based and consider the most recent and effective diagnostic practices.
        6. The output should include diagnostic and clinical decision support information.
        7. In "Diagnoses", only output the diagnosis itself (exact French name from the list). Place all other explanations and analyses (if any) into "Explanations of diagnose".
        8. If additional information is needed for accurate diagnosis, suggest relevant follow-up questions focusing on: symptom characteristics, temporal patterns, aggravating/relieving factors, associated symptoms, medical history.
        9. Provide evidence-based clinical reasoning for the diagnosis.
        10. The output should follow this structured format:
        

    ### Diagnoses
    1. **Diagnosis**: [Exact disease name from the DDXPlus list]
    2. **Explanations of diagnose**: [Clinical reasoning, key symptoms that support this diagnosis, differential considerations]
    
    ### Differential Diagnosis Considerations
    1. **Primary Diagnosis Confidence**: [High/Medium/Low]
    2. **Alternative Diagnoses**: [List 2-3 alternative diagnoses if applicable, with brief reasoning]
    
    ### Instructive Questions for Further Evaluation
    1. **Questions**: [Specific questions to clarify symptoms, timing, severity, or distinguish between similar conditions]
    
    ### Clinical Recommendations
    1. **Immediate Actions**: [Any urgent evaluations or interventions needed]
    2. **Diagnostic Tests**: [Recommended laboratory or imaging studies]
    3. **Treatment Approach**: [Initial management recommendations based on the diagnosis]

    ### Recommendations for Further Evaluations
    1. **Specialist Referrals**: [If applicable]
    2. **Follow-up Timeline**: [Recommended follow-up schedule]
    '''


def generate_diagnosis_report(path, query, retrieved_documents, i,top_n,match_n,model):
    system_prompt_RAGKG = get_system_prompt_for_RAGKG()
    system_prompt=system_prompt_RAGKG
    additional_info= get_additional_info_from_level_2(i ,path,top_n=top_n,match_n=match_n)

    prompt = f"{query}\nRetrieved Documents: {retrieved_documents}\nInformation from knowledge graph about relevant diagnoses, if you think the patient's disease is relevant from the suggestions provided by the atlas please refer to thoses details to distinguish similar diagnoses : {additional_info} .Now complete the tasks in that format"


    ############################################################################################openai
    if model =='gpt-4o' or 'gpt-4o-mini' or 'gpt-3.5-turbo-0125':
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        prompt=f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"""
        LLMclient = InferenceClient(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Meta-Llama-3.1-70B-Instruct",
            # "meta-llama/Llama-2-13b-hf",
            # "Qwen/Qwen2-7B-Instruct",
            # "Qwen/Qwen2.5-0.5B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            token=hf_token
        )
        response = LLMclient.text_generation(prompt=prompt,max_new_tokens=400)
        return response

def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results,
                      columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Original Diagnosis'])
    df.to_csv(output_file, index=False)


folder_path = ob_path
documents = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file_name))]

document_embeddings_file_path='./Embeddings_saved/DDXPlus_document_embeddings.npy'

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path)
if os.path.exists(document_embeddings_file_path):
    document_embeddings = load_embeddings(document_embeddings_file_path)
else:
    document_embeddings = get_embeddings(documents)
    save_embeddings(document_embeddings, document_embeddings_file_path)
