JAM : T5-Based Question Answering System
==================================

Project Overview
----------------

This project implements a **T5-based Question Answering System** using the **Hugging Face Transformers library** and **Wikipedia-sourced data**. The model is trained to answer factual questions using text data extracted from Wikipedia. It utilizes the **T5 Transformer model** to generate answers based on user queries.

Goal
----

The primary objective of this project is to:

-   Train a **T5-based language model** to generate answers to factual questions.

-   Utilize **Wikipedia** as a source of knowledge to create question-answer pairs for training.

-   Develop an **interactive chatbot** that allows users to ask questions and receive AI-generated responses.

-   Implement an **efficient training pipeline** to improve performance and response accuracy.

Features
--------

-   **Automated Wikipedia-based dataset generation**: The system scrapes Wikipedia articles and formulates QA pairs from them.

-   **T5 Model for Answer Generation**: Uses a **pre-trained T5 model** fine-tuned for answering general knowledge questions.

-   **Training Pipeline**: Implements data preprocessing, model fine-tuning, and evaluation.

-   **Interactive QA Mode**: Users can ask questions in real-time after training the model.

-   **Error Handling**: Checks for missing dependencies and ensures compatibility with required libraries.

Current Progress
----------------

### ✅ Implemented:

-   **Data Collection**: Extracts Wikipedia content and formulates QA pairs.

-   **Model Loading and Training**: Initializes and trains a **T5-small model**.

-   **Inference Functionality**: Generates answers for input questions.

-   **Interactive Chat Mode**: Allows users to ask questions and get AI-generated answers.

-   **Model Saving & Loading**: Trained models are saved and reloaded for inference.

### ⏳ Work in Progress:

-   **Fine-tuning and Hyperparameter Optimization**: Improving response quality.

-   **Dataset Expansion**: Adding more Wikipedia articles and diverse topics.

-   **Deployment**: Creating an API for external integrations.

Installation & Setup
--------------------

### 1️⃣ Install Dependencies

Ensure Python is installed and then install the necessary packages:

```
pip install torch transformers sentencepiece wikipedia numpy
```

### 2️⃣ Run the Training Script

Execute the training script to fine-tune the model:

```
python train.py
```

### 3️⃣ Ask Questions Interactively

Once training is complete, start the interactive mode:

```
python qa_chatbot.py
```

Then type your questions and get AI-generated answers.

Training Details
----------------

-   **Model**: T5-small

-   **Dataset**: Wikipedia-based QA pairs

-   **Batch Size**: 8

-   **Learning Rate**: 0.0001

-   **Epochs**: 5

-   **Device**: CPU (can be modified for GPU acceleration)

Example Usage
-------------

```
Your question: Who is the president of the United States?
A: The current president of the United States is Joe Biden.
```

Troubleshooting
---------------

### Common Issues & Fixes

1.  **Missing Dependencies**: Ensure all required packages are installed:

    ```
    pip install -r requirements.txt
    ```

2.  **Wikipedia API Errors**: Some Wikipedia pages may cause errors due to disambiguation.

3.  **Slow Performance**: Running on CPU can be slow; consider using a GPU.

Future Enhancements
-------------------

-   **Deploy as an API** (Flask/FastAPI)

-   **Implement a Web Interface**

-   **Improve Model Accuracy** with fine-tuning


Running Options:
----------------
Reduced CPU :    python train.py

Avoid UltraChat:    python train.py --smaller_model

Fast Training:    python train.py --performance_mode power_saver

Maximum Performance:    python train.py --performance_mode performance --model_size small --batch_size 8

Other optimization options:
- For fastest results with high quality: `python train.py --performance_mode performance --batch_size 6`
- Skip UltraChat loading for faster training: add `--datasets casual+squad` 
- For development testing: add `--epochs 2`
- Absolute fastest training (lower quality): `python train.py --performance_mode performance --model_size small --batch_size 12 --datasets casual --epochs 2`

License
-------

This project is open-source and licensed under the MIT License.

* * * * *

🚀 **Contributions are welcome!** If you have suggestions or improvements, feel free to submit a pull request.