# Machine_Learning_NLP_Project

# 🧠 News Category Classification using Deep Learning (BERT) and Traditional Machine Learning Approaches

This project focuses on **automatic classification of English news articles** into their correct categories using both **Deep Learning (Transformer-based models)** and **Traditional Machine Learning models**.  
The goal is to analyze how modern NLP models such as **BERT** perform compared to classical algorithms like **Logistic Regression** and **Linear SVM** when applied to text-based datasets.

Accurately classifying news articles helps improve **content recommendation systems**, **news aggregation**, and **information retrieval**, enabling readers and companies to efficiently manage and categorize large volumes of information.

---

## 📂 Datasets

### 🧾 *Dataset 1 – English LPC (Long Paragraph Classification)*
**File Name:** `English_train_lpc.xlsx`  
**Source:** [L3Cube Pune IndicNLP GitHub Repository](https://github.com/l3cube-pune/indic-nlp/tree/main)  
**Language Used:** English  
**Size:** ~36,877 training samples + validation/test splits  
**Classes (12 total):** Auto, Business, Education, Elections, Entertainment, Health, India, Lifestyle, Science, Sports, Technology, World  

**Preprocessing Performed:**
- Removal of null rows and unwanted whitespace  
- Label encoding of category column  
- Text tokenization using BERT tokenizer (`bert-base-uncased`)  
- Padding/truncation to fixed sequence length (256 tokens)

---

### 🧾 *Dataset 2 – BBC News Dataset*
**File Name:** `bbc_data.xlsx`  
**Source:** Extracted from open-source BBC news text dataset  
**Size:** ~2,200 records across 5 classes — Business, Entertainment, Politics, Sports, and Tech  
**Preprocessing Performed:**
- Text normalization and cleaning  
- Removal of punctuation, special characters, and URLs  
- TF-IDF vectorization (for traditional ML models)  

---

## ⚙️ Methods

### 🎯 *Problem Approach*
The goal of this study is to predict the correct **news category** for a given article using two different paradigms:
1. **Deep Learning (Transformer Models)**
   - Fine-tuning `bert-base-uncased` on both datasets
   - Tokenizing text sequences and applying classification heads
2. **Traditional Machine Learning**
   - TF-IDF feature extraction followed by classical ML classifiers  
   - Algorithms: Logistic Regression, Linear SVM  

---

### 🧩 *Why This Approach Works*
- **BERT (Bidirectional Encoder Representations from Transformers)** captures the deep semantic and contextual relationships between words, giving it a strong edge for long text understanding.  
- **TF-IDF with Logistic Regression/SVM** serves as a lightweight, interpretable, and efficient baseline for comparison.  
- By comparing both approaches, we can evaluate the trade-offs between computational complexity and predictive power.

---

### 🧭 *Methodology Flow Diagram*

#### Deep Learning (BERT)
