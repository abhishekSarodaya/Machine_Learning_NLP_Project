# Machine_Learning_NLP_Project

# 🧠 News Category Classification using Deep Learning (BERT) and Traditional Machine Learning Approaches

This project focuses on **automatic classification of English news articles** into their correct categories using both **Deep Learning (Transformer-based models)** and **Traditional Machine Learning models**.  
The goal is to analyze how modern NLP models such as **BERT** perform compared to classical algorithms like **Logistic Regression** and **Linear SVM** when applied to text-based datasets.

Accurately classifying news articles helps improve **content recommendation systems**, **news aggregation**, and **information retrieval**, enabling readers and companies to efficiently manage and categorize large volumes of information.

---

## 📂 Datasets

### 🧾 *Dataset 1 – English LPC (Long Paragraph Classification)*
**File Name:** `English_train_lpc.xlsx`  
**Source:** [L3Cube Pune IndicNLP GitHub Repository](https://github.com/l3cube-pune/indic-nlp/tree/main/L3Cube-IndicNews/English/LPC)  
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
**Source:** [BBC articles fulltext and category](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)
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


---

## 🚀 Steps to Run the Code

### 💻 Option 1: Run on Google Colab
1. Upload the `.ipynb` files to your Google Drive.
2. Open each notebook:
   - `Mini_Project_Machine_Learning.ipynb` → LPC dataset (BERT)
   - `ml_mini_part_2.ipynb` → BBC dataset (BERT)
   - `ml_mini_traditional.ipynb` → Classical ML (both datasets)
3. Mount your Google Drive inside Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

# 📊 Experiments and Results Summary

| **Dataset** | **Model** | **Accuracy** | **F1-Score** | **Remarks** |
|--------------|-----------|---------------|---------------|-------------|
| **English LPC** | BERT-base-uncased | ~74 % | ~0.73 | Strong baseline for long-text classification |
| **BBC Dataset** | BERT-base-uncased | ~94 % | ~0.93 | Excellent performance on clean short-text data |
| **English LPC** | Logistic Regression | ~73 % | ~0.72 | Best traditional model for LPC |
| **BBC Dataset** | Logistic Regression | ~98 % | ~0.97 | Matches deep learning accuracy with simpler model |
| **English LPC** | Linear SVM | ~72 % | ~0.72 | Performs slightly below BERT but remains stable for long-text data |
| **BBC Dataset** | Linear SVM | ~98 % | ~0.95 | Outperforms other models; extremely high precision on structured short news articles |


---

### 🔍 Key Observations

- 🧠 **BERT** models achieve strong contextual accuracy for **long and complex articles** (LPC dataset).  
- ⚡ **Logistic Regression** and **Linear SVM** perform extremely well on **structured, shorter text** (BBC dataset).  
- 🕒 **Classical ML models** train faster and require far fewer computational resources than transformer models.  
- 🔄 For **production-grade systems**, a hybrid model that combines **TF-IDF features** with **BERT embeddings** could provide optimal efficiency and accuracy.

---

# 🧾 Conclusion

This project successfully demonstrates **news article classification** using both **Transformer-based Deep Learning** and **Traditional Machine Learning** approaches.

- **BERT** excels at capturing **semantic nuances** in long and context-rich text, making it ideal for complex news corpora such as LPC.  
- **Logistic Regression** and **SVM**, despite their simplicity, perform nearly on par for structured datasets like BBC, proving the continued value of classical ML methods.  
- The experiments validate that **classic ML techniques remain highly competitive** when combined with strong text preprocessing and TF-IDF vectorization.

### 🚀 Future Work
- Experiment with **Indic-BERT**, **DistilBERT**, and **multilingual datasets** for cross-language expansion.  
- Deploy as a **Streamlit or Flask web app** for real-time user predictions.  
- Extend to **multi-label classification** (e.g., articles belonging to multiple topics) and **fake news detection** for greater applicability.

---

# 📚 References

- [L3Cube Pune IndicNLP Repository](https://github.com/l3cube-pune/indic-nlp/tree/main)  
- [BBC News Dataset – Kaggle Source](https://www.kaggle.com/datasets/pariza/bbc-news-summary)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [TF-IDF Text Representation — Stanford NLP Guide](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-term-weighting-1.html)

---

> 🧩 *These experiments underline the power of combining modern transformer architectures with traditional NLP feature-based methods, creating a robust and efficient solution for automated news categorization.*


