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

<h1 align="center">🫁 Hybrid Explainable AI (XAI) Framework for Automatic Tuberculosis Detection</h1>

<p align="center">
  <img src="https://i.postimg.cc/tTTFpzdM/demo.png" alt="Hybrid TB Detection XAI Cover Page">
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/ReportLab-PDF%20Report-success?style=for-the-badge">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/Shariar-Ahamed/hybrid-tb-detection-xai">
  <img src="https://img.shields.io/github/repo-size/Shariar-Ahamed/hybrid-tb-detection-xai">
  <img src="https://img.shields.io/github/last-commit/Shariar-Ahamed/hybrid-tb-detection-xai">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Shariar-Ahamed/hybrid-tb-detection-xai?style=social">
  <img src="https://img.shields.io/github/forks/Shariar-Ahamed/hybrid-tb-detection-xai?style=social">
</p>

<p align="center">
  <a href="https://wakatime.com/badge/user/c7433bc5-6f12-4c97-baea-430790fa608c/project/a814d58e-60f5-4f29-a8d5-c061501de0cb"><img src="https://wakatime.com/badge/user/c7433bc5-6f12-4c97-baea-430790fa608c/project/a814d58e-60f5-4f29-a8d5-c061501de0cb.svg" alt="wakatime"></a>
</p>

---

## 📝 Project Overview

This repository contains the official implementation of the **Hybrid Explainable AI Framework for Automatic Tuberculosis Detection from Chest X-Ray Images Using Convolutional Neural Networks (CNNs)**. This is a final year design project (FYDP) submitted to **Daffodil International University (DIU)**.

The framework utilizes a hybrid CNN model (incorporating **DenseNet121** and **CBAM Attention Blocks**) to classify tuberculosis from chest scans and provides real-time model interpretability using four advanced XAI methods to increase clinical trust.

---

## 📊 Features

* **Hybrid CNN-Attention Architecture**: Combines features from DenseNet121 with Channel and Spatial Attention Blocks (CBAM) to focus on relevant pathological regions.
* **Comparative XAI Engine**: Real-time visualization using four distinct methods:
  * **Grad-CAM**: Visualizes activation heatmaps highlighting general disease regions.
  * **Grad-CAM++**: Focuses on more localized multiple instances of features.
  * **Guided Backpropagation**: Highlights fine-grained, high-frequency textural borders and details.
  * **Saliency Maps**: Blends standard pixel gradients using `COLORMAP_HOT` overlays.
* **Streamlit Dashboard**: Dark-themed, modern glassmorphic dashboard for diagnostic execution.
* **Clinical PDF Report**: Generates print-ready PDFs directly in-memory featuring metadata, confidence metrics, and selected XAI maps.
* **Outstanding Performance**: Achieved **100% test accuracy and sensitivity** on the test subset (630 images) of the hybrid dataset (4,200 chest X-ray scans).

---

## 🔗 Live Demos & Links

* **Hugging Face Space (Project Home)**: [View Space](https://huggingface.co/spaces/shariar-ahamed/hybrid-tuberculosis-detection-xai)
* **Live Web App**: [Live View](https://shariar-ahamed-hybrid-tuberculosis-detection-xai.hf.space/)
* **Streamlit Community Cloud App**: [Live View](https://hybrid-tuberculosis-detection-xai.streamlit.app/)

---

## 👥 Project Team & Supervision

* **Supervised by**: 
  * **Dr. Md. Ali Hossain** (Associate Professor, Department of CSE, DIU)
  * **Mr. Md. Mizanur Rahman** (Co-Supervisor)
* **Developed by**:
  * **Shariar Ahamed Ripon** (ID: 0242310005101019)
  * **Md. Moniruzzaman Rifat** (ID: 0242310005101020)

---

> [!WARNING]
> **Medical Disclaimer:** This dashboard is an explainable AI research prototype for educational and evaluation purposes. All clinical decisions and AI-generated visualization overlays must be verified and validated by a certified radiologist before medical action is taken.
