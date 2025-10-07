# LLM Fake News Detection Algorithm

This project develops and evaluates machine learning and large language model (LLM) methods to classify online news articles as real or fake. It begins with a baseline **logistic regression** model using TF-IDF features, then extends the analysis with **LLM-based contextual classification** to recover misclassified cases and assess model complementarity.

---

## Overview

The project compares traditional text classification techniques with LLM-driven approaches for fake news detection.  
In **Step 1**, a logistic regression model is trained on balanced “True” and “Fake” news datasets using TF-IDF representations, achieving near-perfect accuracy (98.9%) on held-out data.  
In **Step 2**, a GPT-4o-mini model is introduced to reclassify the most challenging cases — those misclassified by the baseline model — using structured, Pydantic-validated prompts.

The LLM achieved a **64% accuracy** on this difficult subset, substantially improving over the baseline’s 0% accuracy on the same examples. When filtering predictions with **≥0.9 confidence**, the model reached **95% accuracy**, demonstrating that confidence scores can serve as a meaningful indicator of prediction reliability.

---

## Data

The dataset used for this project comes from Kaggle’s **Fake and Real News Dataset**, which contains over 40,000 labeled news articles.  
It can be accessed directly here:

[Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)


## Skills Used

- Classical machine learning (Logistic Regression, TF-IDF feature extraction)  
- LLM-based text classification (GPT-4o-mini)  
- Natural Language Processing (NLTK, text cleaning, stopword removal)  
- Explainable AI (SHAP feature importance analysis)  
- Model evaluation and visualization (confusion matrices, accuracy, F1-scores)  
- Structured model querying with **Pydantic** and **OpenAI API integration**  
- Data manipulation and visualization in **Python** (`pandas`, `scikit-learn`, `seaborn`, `matplotlib`)

---

## Contributions

- **Built** a reproducible fake news detection pipeline integrating traditional and LLM-based classifiers.  
- **Trained** a logistic regression baseline using TF-IDF vectors achieving 98.9% accuracy.  
- **Visualized** model interpretability with SHAP, identifying linguistic markers of credibility versus sensationalism.  
- **Developed** an LLM-based validation layer to reclassify baseline misclassifications, improving performance by 64% on the hardest examples.  
- **Introduced** confidence-based filtering, showing that high-confidence LLM predictions reach 95% accuracy.  
- **Proposed** a hybrid classification framework combining efficient linear models with LLMs for ambiguous or low-confidence cases.

---

