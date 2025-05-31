# A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning





## Appendix A. Doctor Evaluation Framework

To comprehensively evaluate the performance of MedRAG, we engaged experienced medical professionals to assess the system's responses using the QUEST (Quality, Understanding, Expression, Safety, and Trust) evaluation framework proposed by Tam et al. [1]. This framework provides a multi-dimensional human evaluation of medical LLM responses across five key dimensions:

1. **Quality of Information**: Evaluates the accuracy, relevance, currency, comprehensiveness, consistency, agreement, and usefulness of the information provided in the responses.

2. **Understanding and Reasoning**: Assesses the system's ability to comprehend medical queries and demonstrate logical reasoning in formulating responses.

3. **Expression Style and Persona**: Measures the clarity and empathy in the response style, ensuring effective communication with healthcare professionals and patients.

4. **Safety and Harm**: Examines critical safety aspects including bias, potential harm, self-awareness, and the presence of fabrication, falsification, or plagiarism in responses.

5. **Trust and Confidence**: Evaluates the level of trust and satisfaction that users derive from the system's responses.

For the evaluation process, we selected a subset of test cases from our dataset along with their corresponding LLM-generated answers. Doctors were then asked to evaluate these responses using the QUEST framework criteria scored on a scale of 1 to 5. The evaluation results are presented in Figure, which illustrates the doctors' assessments across different dimensions of the framework.

[1] Tam T Y C, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review[J]. NPJ digital medicine, 2024, 7(1): 258.




## Appendix B. End-to-End Evaluation of Speech-to-Text Integration

To assess the practical applicability of MedRAG in real clinical settings, we conducted an end-to-end evaluation of the speech-to-text (STT) integration module. This evaluation focused on simulating realistic doctor-patient conversations and measuring the system's ability to process and understand spoken medical information.

We utilized GPT-4 to transform structured EHRs into natural, multi-turn dialogues between doctors and patients. The generated dialogues covered various medical scenarios and included typical clinical terminology and expressions. These simulated dialogues were then converted to speech input, processed by the STT module, and fed into MedRAG's LLM component for final disease diagnosis accuracy evaluation. The results are shown in Figure.

<div align="center"> <img src="./Fig/Modal_eval.png" alt="clustering" width="500"> </div >
    <p><em>The result of different modalities in CPDD.</em></p >


The evaluation results revealed that the STT module successfully captured the majority of medical terminology while preserving some natural language variations and speech patterns. The LLM component demonstrated robust error tolerance, maintaining reasonable diagnostic accuracy despite potential transcription errors. The final output maintained clinical relevance while preserving some characteristic "robotic" patterns. Notably, the system maintained consistent diagnostic accuracy regardless of the input format (text vs. speech), suggesting that the LLM's strong language understanding capabilities effectively compensated for any STT-related errors. It's important to note that this evaluation focused solely on disease diagnosis accuracy, independent of the STT module's performance metrics. The results demonstrate that MedRAG can effectively process and understand spoken medical information while maintaining its diagnostic capabilities.

