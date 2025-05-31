# A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning





## Appendix A. Doctor Evaluation Framework

To comprehensively evaluate the performance of MedRAG, we engaged experienced medical professionals to assess the system's responses using a comprehensive evaluation framework. This framework provides a multi-dimensional human evaluation of medical LLM responses across five key dimensions:

| **Dimension**               | **Description** |
|----------------------------|-----------------|
| **Trust**                  | Assesses the extent to which the clinician perceives the system's diagnostic suggestions as credible, reliable, and clinically appropriate. |
| **Adoption Intention**     | Measures the physician’s willingness to adopt the system's output as a basis for clinical decision-making in their current practice. |
| **Future Use Likelihood**  | Evaluates the clinician’s intention or likelihood to continue using the system in future clinical scenarios or similar cases. |
| **Recommendation Willingness** | Captures the user’s inclination to recommend the system to colleagues or peers, reflecting perceived overall value and utility. |
| **Reliance Level**         | Determines the degree to which clinicians are comfortable relying on the system’s outputs without the need for additional verification or manual cross-checking. |

Each dimension is rated on a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree) based on physicians’ direct interaction with system outputs.

For the evaluation process, we selected three test cases from our dataset along with their corresponding MedRAG generated answers to compare with GPT-4o generated answers. Doctors were then asked to evaluate these responses using this framework criteria scored on a scale of 1 to 5. The evaluation results are presented in Figure, which illustrates the doctors' assessments across different dimensions of the framework.



## Appendix B. End-to-End Evaluation of Speech-to-Text Integration

To assess the practical applicability of MedRAG in real clinical settings, we conducted an end-to-end evaluation of the speech-to-text (STT) integration module. This evaluation focused on simulating realistic doctor-patient conversations and measuring the system's ability to process and understand spoken medical information.

We utilized GPT-4 to transform structured EHRs into natural, multi-turn dialogues between doctors and patients. The generated dialogues covered various medical scenarios and included typical clinical terminology and expressions. These simulated dialogues were then converted to speech input, processed by the STT module, and fed into MedRAG's LLM component for final disease diagnosis accuracy evaluation. The results are shown in Figure.

<div align="center"> <img src="./Fig/Modal_eval.png" alt="clustering" width="400"> </div >
    <p><em>The result of different modalities in CPDD.</em></p >


The evaluation results revealed that the STT module successfully captured the majority of medical terminology while preserving some natural language variations and speech patterns. The LLM component demonstrated robust error tolerance, maintaining reasonable diagnostic accuracy despite potential transcription errors. Notably, the system maintained consistent diagnostic accuracy regardless of the input format (text vs. speech), suggesting that the LLM's strong language understanding capabilities effectively compensated for any STT-related errors. It's important to note that this evaluation focused solely on disease diagnosis accuracy, independent of the STT module's performance metrics. The results demonstrate that MedRAG can effectively process and understand spoken medical information while maintaining its diagnostic capabilities.

