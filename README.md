---
title: Hybrid Tuberculosis Detection XAI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
---

# 🫁 Hybrid Explainable AI (XAI) Framework for Automatic Tuberculosis Detection

This repository contains the official implementation of the **Hybrid Explainable AI Framework for Automatic Tuberculosis Detection from Chest X-Ray Images Using Convolutional Neural Networks (CNNs)**.

This is a final year design project (FYDP) submitted to **Daffodil International University (DIU)**.

## 🚀 Live Demos
You can access the interactive dashboard using the links below:
* **Hugging Face Spaces**: [Link](https://huggingface.co/spaces/shariar-ahamed/hybrid-tuberculosis-detection-xai)
* **Streamlit Community Cloud**: [Link](https://hybrid-tuberculosis-detection-xai.streamlit.app/)

## 📊 Project Features
* **Hybrid CNN-Attention Architecture**: Integrates DenseNet121 with Channel and Spatial Attention Blocks (CBAM) to focus on relevant pathological regions.
* **Explainable AI (XAI)**: Utilizes **Grad-CAM** to generate real-time activation heatmaps on Chest X-Rays, explaining exactly *why* the model made its decision.
* **High Sensitivity & Accuracy**: Achieved **100% test accuracy and sensitivity** on the test subset (630 images) of the hybrid dataset (4,200 chest X-ray scans).
* **Interactive Streamlit Dashboard**: User-friendly UI built with rich styling, dark mode glassmorphic UI, real-time prediction confidence score, and downloadable heatmap visualizations.

## 👥 Project Team & Supervision
* **Supervised by**: 
  * **Dr. Md. Ali Hossain** (Associate Professor, CSE, DIU)
  * **Mr. Md. Mizanur Rahman** (Co-Supervisor)
* **Developed by**:
  * **Shariar Ahamed Ripon** (ID: 0242310005101019)
  * **Md. Moniruzzaman Rifat** (ID: 0242310005101020)

---
*Medical Disclaimer: This dashboard is a research prototype for educational and evaluation purposes. All clinical decisions must be verified by a certified radiologist.*
