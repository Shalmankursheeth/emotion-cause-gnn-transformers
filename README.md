# Multi-Modal Emotion and Cause Detection in Tweets using Graph Neural Networks and Transformer Architectures

This project focuses on abstractive generation and classification of emotions and their causes from tweets. It leverages the synergy between Graph Neural Networks (GNNs) and Transformer-based architectures to perform multi-modal analysis of textual data. The primary objective is to detect emotions and generate their underlying causes in a multi-class emotion detection task.

# 🔍 Problem Statement
Understanding not just what emotion a tweet expresses but why that emotion is expressed is a crucial task in emotion-aware systems. This project explores how advanced architectures can extract semantic relationships and abstractly generate cause phrases to explain the detected emotions.

📁 Repository Structure

# 📦 Multi-Modal-Emotion-and-Cause-Detection

├── approach1_graph_based.py             # Graph-based Emotion-Cause classification
├── approach2_transformers_combined.py  # Transformer-based dual-stage pipeline
├── approach3_tree_bart.py              # Tree-BART for abstractive cause generation
├── data_sample.csv                     # Sample dataset (10 rows for demo/training)
├── README.md                           # Project documentation 
⚠️ Note: Model weights and large classifier files are excluded to keep the repository lightweight. You can train and fine-tune models using the provided code and your own environment.

# 🧠 Approaches Used
✅ 1. Graph-Based Classification (approach1_graph_based.py)
Utilizes Graph Neural Networks to understand relationships between words/entities.

Performs emotion classification using node embeddings from the graph.

Suitable for scenarios where syntactic structure adds context.

✅ 2. Transformer-Based Dual Stage (approach2_transformers_combined.py)
Stage 1: BART model generates an abstractive cause for the given tweet.

Stage 2: RoBERTa or similar transformer classifies the emotion using both tweet and generated cause.

Supports fine-tuning with HuggingFace libraries.

✅ 3. Tree-BART for Cause Generation (approach3_tree_bart.py)
Enhances BART with tree-based dependency parsing for structured abstractive generation.

Improves cause generation quality by integrating linguistic structure.

# 📊 Dataset
Format: .csv with columns like Tweet, Emotion, Cause (for training).

Includes 10 sample rows (data_sample.csv) for testing and demonstration.

You can use your own larger dataset to train and evaluate models.

Due to space limitations, a full dataset is not hosted here. Publicly available datasets (like EmotionX, GoEmotions, or custom annotated Twitter data**) can be used.

# ⚙️ Setup and Usage
📦 Install Requirements
Requirements depend on the approach (e.g., PyTorch, Transformers, NetworkX, DGL, spaCy, etc.).

🚀 Run Code

# Run graph-based classifier
    python approach1_graph_based.py

# Run transformer dual-stage classifier
    python approach2_transformers_combined.py

# Run Tree-BART based generator
    python approach3_tree_bart.py
🏁 Expected Output
Cause Generation: Given a tweet, generate a plausible abstract cause phrase.

Emotion Classification: Predict the corresponding emotion class (e.g., joy, sadness, anger, etc.).

Multimodal Reasoning: Combine linguistic graph and transformer attention for explainable emotion detection.

# 💡 Applications
Mental health monitoring on social media.

Chatbot empathy training.

Customer feedback analysis.

Emotion-aware virtual assistants.


# 🙌 Acknowledgements
Inspired by research in multi-modal emotion recognition, transformer-based NLP, and semantic cause-effect modeling.

Built as part of a research initiative at NIT Trichy.

