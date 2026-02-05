# 🏠 Real-World Digit Recognition (SVHN) using Deep Learning

### 🎯 Project Overview (Executive Summary)

This project focuses on the automated recognition of house numbers from Google Street View images (SVHN dataset). The objective was to build a robust machine learning model capable of identifying digits in noisy, real-world environments with high precision.

This technology is a fundamental component for automated mapping systems, logistics, and digitizing the physical world.

### 🏆 Key Results

* **Final Accuracy:** Achieved **91.7% accuracy** on the unseen test dataset.
* **Improvement:** Transitioning from a standard Artificial Neural Network (ANN) to a Convolutional Neural Network (CNN) yielded a **~17% accuracy boost**.
* **Optimization:** Successfully mitigated overfitting issues by implementing regularization techniques such as **Dropout** and **Batch Normalization**.

---

### 🛠️ Tech Stack & Tools

* **Language:** Python 3
* **Deep Learning:** TensorFlow, Keras (Sequential API)
* **Data Science:** NumPy, Pandas, H5py
* **Visualization:** Matplotlib, Seaborn
* **Key Concepts:** CNN, ANN, Data Normalization, One-Hot Encoding, Hyperparameter Tuning

---

### 🧠 Methodology & Approach

**1. Data Analysis & Preparation**

* Handled a dataset of 42,000 training images and 18,000 test images (32x32 pixels).
* Applied pixel normalization (scaling 0-255 to 0-1) to ensure faster model convergence.
* Converted target variables into One-Hot Encoded vectors for classification.

**2. Model Development (Iterative Process)**

* **Model A (Baseline - ANN):** A simple Feed-Forward Neural Network.
* *Result:* Reached only ~75% accuracy. Failed to capture spatial patterns effectively.


* **Model B (Basic CNN):** Standard Convolutional Network.
* *Result:* High training accuracy (~98%) but lower validation accuracy (~85%). Indicated severe **overfitting**.


* **Model C (Optimized CNN - Final):**
* Integrated **BatchNormalization** layers to stabilize learning.
* Implemented **Dropout (0.5)** to prevent overfitting.
* Utilized **LeakyReLU** activation for better gradient flow.
* *Result:* A stable model with **91.7% accuracy** and excellent generalization on new data.



**3. Evaluation**

* Utilized **Confusion Matrix** to analyze specific misclassifications (e.g., distinguishing between visually similar digits like 5 and 6).
* Evaluated Precision, Recall, and F1-Scores across all classes.

---

### 📊 Results Snapshot

> "By deploying an optimized CNN architecture, the model error rate was minimized, ensuring reliable digit reading even in cases where digits are visually similar."
