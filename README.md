# Deep Learning Projects - Continuous Assessment

### 👤 Name: Purva Baghel
### 🆔 PRN: 22070521130
### 🧑‍🎓 Semester: 6  

---

## 📘 Case Study Title:
### Continuous Assessment - Introduction to Deep Learning

---

## 🧠 Introduction

This project showcases the application of deep learning techniques in solving real-world classification problems using artificial neural networks (ANNs) and convolutional neural networks (CNNs). The work is divided into two parts:

1. **Handwritten Digit Recognition using CNNs**
2. **Heart Disease Prediction using ANNs**

Each problem reflects a vital domain of deep learning: computer vision and medical diagnostics. The aim is to explore how deep learning models can automate, enhance, and simplify complex tasks with high accuracy and efficiency.

---

## 🔍 Problem Statement

As the digital world expands, the ability to process and analyze massive amounts of unstructured data has become vital. This project aims to address:

- **The challenge of recognizing handwritten digits** from images in a reliable and automated way.
- **Predicting the presence of heart disease** based on various medical indicators using ANN.

Both problems are classification tasks and are critical in their respective domains – from digitizing documents to assisting doctors in medical diagnoses.

---

## 📌 Project 1: Handwritten Digit Recognition with Deep Learning

### 📂 Description:

This part of the project focuses on implementing a Convolutional Neural Network (CNN) model using the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0–9). CNNs are chosen due to their proven effectiveness in image-based tasks.

### 🎯 Objective:

To build a deep learning model that can accurately classify digits from images.

### 🛠️ Steps Implemented:

- **Data Loading and Preprocessing**: 
  - Imported the MNIST dataset.
  - Normalized pixel values for optimal learning.
  - Reshaped data to fit CNN input format.

- **Model Architecture**: 
  - Used multiple convolutional layers with ReLU activation.
  - Applied max pooling to reduce spatial dimensions.
  - Flattened and connected to dense layers for classification.

- **Training**: 
  - Loss Function: Categorical Crossentropy.
  - Optimizer: Adam.
  - Metrics: Accuracy.

- **Evaluation**: 
  - Evaluated performance on test dataset.
  - Accuracy achieved: ~98%.

- **Prediction and Visualization**: 
  - Predicted digits from test images.
  - Visualized results using `matplotlib`.

### ✅ Output:

- Achieved excellent accuracy on test set.
- Model successfully recognized handwritten digits.
- Visual output confirmed model's prediction capabilities.

---

## 📌 Project 2: Heart Disease Prediction Using Artificial Neural Networks

### 📂 Description:

This part targets structured data to predict whether a person is likely to develop heart disease based on clinical features. The dataset includes attributes such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more.

### 🎯 Objective:

To predict the presence of heart disease using an artificial neural network trained on medical data.

### 🛠️ Steps Implemented:

- **Data Loading and Cleaning**: 
  - Loaded dataset using Pandas.
  - Removed missing or invalid entries.
  - Applied feature scaling.

- **Feature Engineering**:
  - Used One-Hot Encoding for categorical variables.
  - Separated features and target labels.

- **Model Architecture**:
  - Built a Feed-Forward Neural Network.
  - Used ReLU and Sigmoid activations in layers.
  - Multiple Dense layers for learning.

- **Training**:
  - Loss Function: Binary Crossentropy.
  - Optimizer: Adam.
  - Evaluated accuracy per epoch.

- **Evaluation**:
  - Accuracy and Loss plots.
  - Confusion Matrix to assess prediction quality.

- **Prediction**:
  - Model tested on custom input values.
  - Displayed whether a patient is likely to have heart disease.

### ✅ Output:

- Achieved ~85-90% prediction accuracy.
- Reliable and robust classification performance.
- Visualization tools provided deep insight into model behavior.

---

## 🎯 Why Solve These Problems?

- **Digit recognition** automates data entry and verification in finance, postal systems, and academics.
- **Heart disease prediction** aids in early diagnosis, potentially saving lives and reducing healthcare burdens.

---

## 🧩 Technologies Used

- Python  
- TensorFlow & Keras  
- Matplotlib  
- Pandas & NumPy  
- Scikit-learn  

---

## 📎 Conclusion

These projects demonstrate the real-world impact of deep learning. While the digit recognition system highlights the strength of CNNs in computer vision, the heart disease model showcases ANN’s capability in structured data classification. Both models are stepping stones toward more complex deep learning solutions in healthcare, automation, and beyond.

---

