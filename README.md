# Banglish Emotion Detection – Multi-Label Classification

This project focuses on **Banglish (Bengali–English code-mixed) Emotion Classification** using transformer-based deep learning models.  
We classify text into **7 different emotions** using a combination of **pre-existing datasets** and **newly collected data**.

---

## 🧩 Emotion Labels (7 Classes)

1. Love
2. Sad
3. Angry
4. Surprise
5. Fear
6. Joy
7. Hate

---

## 📊 Dataset Overview

| Dataset Component | Samples | Annotation |
|------------------|----------|------------|
| Pre-existing Dataset | **15,648** |  *No annotation provided* |
| Collected Dataset | **11,608** | ✔ Annotated by author |
| **Total Dataset** | **27,256** | ✔ Fully annotated by author |

### ✔ Important Note  
The **pre-existing dataset originally had NO emotion labels**.  
 manually **annotated all 15,648 samples**.

This is a major contribution:  
→ **Full annotation of a large Banglish dataset (15.6k samples)**  
→ **Additional 11.6k collected + annotated samples**  
→ Total **27k+ fully emotion-labeled data for Banglish text**

---

## 🏆 Contribution of This Work

### 🔹 **1. Full Manual Annotation (26k+ samples)**
- Annotated **15,648 unlabelled pre-existing samples**
- Collected & annotated **11,608 new samples**
- Final dataset is fully labeled into 7 emotions  
- One of the **largest Banglish emotion datasets**


### ✔ Data Sources
- Existing publicly available Banglish / code-mixed emotion datasets  
- Social media comments, posts, and conversations (collected manually & semi-automatically)

### ✔ Data Preprocessing
- Language normalization (Banglish variations normalized)
- Cleaning misspellings, emojis, repeated characters
- Removing URLs, mentions, numbers
- Handling code-mixed text (Bn+En token patterns)
- One-Hot encoding via MultiLabelBinarizer


### 🔹 **2. Multi-Label Classification**
- Real-world Banglish sentences often contain **multiple overlapping emotions**  
  Example: *"amar mathay onek pressure, r kisu bujhtesi na 😞😡"* → *Sad + Angry*

This project solves this using multi-label training.

### 🔹 **3. Model Contribution**
- Implemented multilingual Transformer models  
  - XLM-R /  distilbert-base-multilingual-cased /Multilingual E5 Large Embeddings + XGBoost 
- Trained with class imbalance handling  
- Sigmoid-based multi-label output layer  
- Achieved higher accuracy than baseline datasets

### 🔹 **4. Evaluation Contribution**
- Macro F1, Micro F1  
- ROC-AUC for multi-label emotion prediction  
- Confusion & correlation analysis

---

## 🧠 Model Architecture

- Transformer Encoder Model (HuggingFace)
- Max Sequence Length: 128 / 256
- Optimizer: AdamW
- Learning Rate: 3e-5
- Loss: BCEWithLogitsLoss (for multi-label)
- Metrics:
  - Accuracy  
  - Macro / Micro F1  
  - ROC-AUC  
  - Hamming Loss  

---

## 🚀 Training Setup

### **TrainingArguments**

```python
training_args = TrainingArguments(
    output_dir="ml_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=10,
)
