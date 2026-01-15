# CA6001_Student-Depression
Engineered a resource-efficient depression screening system using Knowledge Distillation in TensorFlow/Keras, optimizing model size and inference speed while maintaining high accuracy.
# Non-Intrusive Student Depression Screening via Knowledge Distillation

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project addresses the challenge of early depression screening in student populations. Traditional methods rely on subjective questionnaires (e.g., PHQ-9) or sensitive personal data. 

**My Solution:** I developed a lightweight, privacy-preserving Deep Learning model that detects depression risk using **only 10 objective behavioral markers** (e.g., Sleep, Diet, Study Hours), achieving performance comparable to a model using 46 comprehensive features. 

This was achieved via **Knowledge Distillation (KD)**, compressing a complex "Teacher" model into a compact "Student" model suitable for non-intrusive edge deployment.

## 🚀 Key Highlights
- **Privacy-First:** Reduced input features from **46** (including sensitive data) to **10** (purely behavioral), preserving user privacy.
- **High Performance:** The distilled student model achieved an **F1-Score of 0.865**, matching the Teacher model (0.867) and significantly outperforming the non-distilled baseline (0.623).
- **Knowledge Transfer:** Utilized a custom loss function combining *Binary Cross-Entropy* (Hard Targets) and *KL Divergence* (Soft Targets) to transfer "dark knowledge" from the teacher.

## 🛠️ Tech Stack & Methods
- **Framework:** TensorFlow / Keras
- **Technique:** Knowledge Distillation (Teacher-Student Architecture)
- **Data Handling:** SMOTE for class balancing, StandardScaler for feature normalization.

## 📊 Model Architecture & Results (Test Set)

| Model Role | Architecture | Input Features | F1-Score | Accuracy | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (Upper Bound)** | **8-8-8-1 NN** | **46** (Full) | **0.867** | **84.2%** | 87.6% |
| Student (Baseline) | 16-8-1 NN | 10 (Behv. only) | 0.623 | 60.8% | 55.3% |
| **Student (Distilled) 🌟** | **16-8-1 NN** | **10** (Behv. only) | **0.865** | **83.8%** | **88.7%** |

> **Result:** Knowledge Distillation improved the F1-Score of the student model by **~39%**, effectively bridging the gap between limited behavioral data and comprehensive psychological profiling.
