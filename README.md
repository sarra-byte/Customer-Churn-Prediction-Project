## Customer Churn Prediction Using Hybrid Neural Networks

This project implements a **customer churn prediction model based on a hybrid neural network architecture**, inspired by the research paper:

 **Liu, X., Xia, G., Zhang, X., Ma, W., & Yu, C. (2024). Customer churn prediction model based on hybrid neural networks. Scientific Reports, 14, 30707.**

## Project Overview

Customer churn prediction aims to identify customers who are likely to stop using a company's products or services. This is an important classification problem because customer behavior can involve complex relationships, long-term dependencies, and short-term behavioral patterns.

This project reproduces and adapts the main methodology proposed in the **CCP-Net** model presented by Liu et al. The proposed architecture combines **Multi-Head Self-Attention (MHSA), Bidirectional LSTM (BiLSTM), and CNN** to learn different types of information from customer data.

## Why a Hybrid Neural Network?

The main idea behind CCP-Net is that a single neural network may not be sufficient to capture all the patterns involved in customer churn.

Customer behavior can contain:

- complex relationships between different features;
- long-term behavioral dependencies;
- local and short-term patterns;
- imbalanced churn and non-churn classes.

Therefore, the authors designed a hybrid architecture in which each component has a specific role.

### 1. Multi-Head Self-Attention (MHSA)

The **Multi-Head Self-Attention** mechanism is used to capture **global dependencies and complex relationships between features**.

Instead of processing the information only step by step, self-attention allows the model to examine different parts of the input and determine which information is more relevant.

The multi-head mechanism also allows the model to learn different relationships in parallel.

In the context of customer churn, this can help identify relationships between different aspects of a customer's historical behavior and consumption patterns.

### 2. Bidirectional LSTM (BiLSTM)

The output of the attention mechanism is then processed by a **Bidirectional Long Short-Term Memory (BiLSTM)** network.

BiLSTM is designed to capture **long-term dependencies in sequential data**.

The bidirectional structure processes the sequence in both directions, allowing the model to obtain a more complete contextual representation.

This is useful for customer churn prediction because churn can be influenced by behavioral patterns that occur over different periods of time.

The LSTM architecture also uses gating mechanisms to control which information should be remembered or forgotten, making it suitable for modeling long-term dependencies.

### 3. Convolutional Neural Network (CNN)

The output of the BiLSTM is then passed to a **Convolutional Neural Network (CNN)**.

While BiLSTM focuses on long-term dependencies, CNN is used to extract **local patterns and short-term behavioral characteristics**.

For example, a sequence may contain a local pattern representing a recent change in customer activity. Convolutional filters can identify such patterns efficiently.

Therefore, CNN complements the global and long-term information learned by the previous modules.

### 4. Why Combine MHSA + BiLSTM + CNN?

The three components have complementary roles:

```text
                 Customer Data
                      │
                      ▼
          Multi-Head Self-Attention
                      │
          Global Dependencies
                      │
                      ▼
                   BiLSTM
                      │
          Long-Term Dependencies
                      │
                      ▼
                    CNN
                      │
            Local Patterns
                      │
                      ▼
             Classification
                      │
                      ▼
              Churn / No Churn
```

The objective is therefore not simply to add three neural networks together, but to combine their different feature-extraction capabilities.

The paper reports that this hybrid architecture improves the ability to model complex customer behavior compared with individual or simpler hybrid architectures. Ablation experiments also showed that removing individual components reduced performance on several datasets.

## Handling Class Imbalance with ADASYN

Customer churn datasets are often imbalanced because the number of customers who leave a service can be considerably smaller than the number of customers who remain.

To address this problem, the original CCP-Net methodology uses **ADASYN (Adaptive Synthetic Sampling)**.

ADASYN generates synthetic samples for the minority class, with greater attention to minority observations that are more difficult to learn.

The objective is to reduce the bias that can occur when training a model on an imbalanced dataset.

## Model Architecture

The complete pipeline used in this project can be summarized as:

```text
Raw Customer Data
       │
       ▼
Data Preprocessing
       │
       ├── Encoding
       ├── Feature Scaling
       └── ADASYN
       │
       ▼
Multi-Head Self-Attention
       │
       ▼
BiLSTM
       │
       ▼
CNN
       │
       ▼
Fully Connected Layer
       │
       ▼
Sigmoid
       │
       ▼
Churn Prediction
```

## Training Configuration

The implementation uses the following training strategy:

- **Problem:** Binary classification
- **Loss Function:** Binary Cross-Entropy 
- **Optimizer:** Adam
- **Output Activation:** Sigmoid
- **Classification Threshold:** 0.5
- **Early Stopping:** Applied during training
- **Cross-Validation:** 10-fold cross-validation

## Datasets

The methodology was evaluated on multiple customer-related datasets from different domains, including:

- Telecom
- Banking
- Insurance
- News

Using several datasets allows the approach to be evaluated across different types of customer behavior and helps investigate its generalization across domains.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics provide a more complete evaluation than accuracy alone, particularly when dealing with imbalanced churn datasets.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Reference

Liu, X., Xia, G., Zhang, X., Ma, W., & Yu, C. (2024).  
**Customer churn prediction model based on hybrid neural networks.**  
*Scientific Reports, 14*, 30707.

DOI: **10.1038/s41598-024-79603-9**

The original article is available through *Scientific Reports*.

## Disclaimer

This repository contains an **academic implementation and adaptation** inspired by the methodology described in the referenced paper. It is not the original implementation released by the authors.
