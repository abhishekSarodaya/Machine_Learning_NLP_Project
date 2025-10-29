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

# ⚙️ Methods

## 🎯 Problem Approach

The general goal of this project is to automatically classify **English news articles** into their correct categories using **Natural Language Processing (NLP)** and **Machine Learning (ML)** methods.  

The experiments were performed on two datasets — **English LPC (Long Paragraph Classification)** and **BBC News** — to evaluate performance differences between **Deep Learning (BERT)** and **Traditional ML** approaches.

To achieve that, the process was broken down into four main stages:

| **Step** | **Description** |
|-----------|-----------------|
| **Data Preprocessing** | Cleaning and normalizing raw news text data to remove noise, HTML tags, punctuation, and irrelevant symbols. Label encoding was used for class mapping. |
| **Feature Extraction (TF-IDF / Tokenization)** | For traditional ML, text was vectorized using *Term Frequency–Inverse Document Frequency (TF-IDF)* to convert text into numerical features. For deep learning, BERT tokenizer was applied to convert text into token IDs and attention masks. |
| **Model Training** | Applied both deep learning and traditional ML algorithms — including BERT fine-tuning, Logistic Regression, and Linear SVM — to classify articles into their correct categories. |
| **Evaluation and Comparison** | Evaluated all models using **Accuracy**, **Precision**, **Recall**, and **F1-Score** to determine the best performing approach for each dataset. |

**Table 1:** Steps involved in News Classification using NLP and Machine Learning Approaches

---

## 💡 Why This Approach Works

- **BERT** captures rich contextual word meanings and long-term dependencies across entire paragraphs, which is ideal for long-form news (English LPC dataset).  
- **TF-IDF** effectively measures word relevance across articles, ensuring frequent but uninformative words (like *“said”*, *“news”*) don’t dominate.  
- **Classical ML models (Logistic Regression, Linear SVM)** are lightweight, easy to interpret, and perform exceptionally well on smaller, clean datasets like BBC News.  
- **NLP preprocessing (cleaning, normalization, and tokenization)** ensures models learn meaningful text features rather than noise.

This hybrid design ensures precision, interpretability, and efficiency — making it practical for real-world media classification or news analytics systems.

---

## 🧩 Alternative Approaches Considered

| **Approach** | **Description** | **Reason for Not Selecting** |
|---------------|------------------|-------------------------------|
| **Deep Neural Networks (LSTM, CNN)** | Can model long-term dependencies and sentence context. | Require large labeled datasets and GPUs for optimal performance. High training cost for minimal accuracy gain in this domain. |
| **Word2Vec / GloVe Embeddings** | Capture semantic relationships between words. | TF-IDF and BERT already provide strong representations; additional embedding layers added little improvement. |
| **Naïve Bayes / Random Forest** | Simpler traditional classifiers. | Performed lower in accuracy and F1 due to independence assumptions and sparse TF-IDF vectors. |

**Table 2:** Comparison of Alternative Approaches and Reasons for Not Selecting Them

---

## 🧭 Methodology Flow Diagram

### Deep Learning Pipeline

