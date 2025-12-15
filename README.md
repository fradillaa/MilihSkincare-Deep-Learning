# MilihSkincare-Deep-Learning

This repository contains a transformer-based skincare product recommendation system that combines **multi-label skin type classification** and **embedding-based content filtering**. The system leverages BERT to learn semantic representations of skincare product descriptions and generate relevant product recommendations without relying on user interaction data.

---

## Project Overview

Selecting suitable skincare products can be challenging due to the large number of available products and complex ingredient information. This project addresses the problem by applying **Natural Language Processing (NLP)** techniques to automatically analyze product descriptions and recommend skincare products based on semantic similarity and skin type compatibility.

Our study focuses on evaluating the skincare landscape, consumer trends, and competition pertinent to the introduction of new skincare products, while proposing a technical solution through transformer-based models.

---

## Methodology Summary

The system adopts a **BERT-based hybrid architecture** with two main learning objectives:

1. **Multi-label Classification**  
   Predicts suitable skin types for each product (e.g., dry, oily, sensitive).

2. **Embedding Learning**  
   Generates dense semantic embeddings for skincare products to support similarity-based recommendation.

The trained embeddings are used in a **content-based filtering** framework to retrieve and rank relevant products.

---

## System Architecture

**Main Components:**
- **BERT Encoder**: Extracts contextual text representations.
- **Classification Head**: Performs multi-label skin type prediction.
- **Embedding Head**: Produces normalized product embeddings for recommendation.
- **Recommendation Engine**: Computes cosine similarity between product embeddings.
- **User Interface**: Interactive frontend implemented using Streamlit.

---

## Implementation Details

### Core Files
- `ModelDeepLearning.ipynb`  
  Contains:
  - Data preprocessing
  - Model architecture definition
  - Model training and evaluation
  - Embedding extraction and storage

- `app.py`  
  Contains:
  - User interface (UI/UX)
  - Model inference
  - Recommendation display logic  
  *(No training or evaluation is performed in this file)*

---

## User Interface

The system is deployed through a Streamlit-based interface that allows users to:
- Provide skincare-related input
- Receive ranked product recommendations
- Interact with the trained model in real time

The UI serves solely as an application layer and does not modify model parameters.

---

## Evaluation

Model performance is evaluated using common **multi-label classification metrics**, including macro-averaged F1-score. Sigmoid activation with a fixed threshold is applied to generate binary predictions.

---

## How to Run the Project

This project consists of two main stages:  
**(1) model training and embedding generation**, and  
**(2) running the user interface application**.

---

### 1. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
