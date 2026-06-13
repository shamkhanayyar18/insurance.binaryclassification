# Insurance Premium Prediction using Deep Learning

This repository contains a Deep Learning project built with TensorFlow and Keras to predict medical insurance cost charges based on individual demographic and health factors.

## 📋 Project Overview
Predicting medical insurance costs helps organizations and individuals estimate future healthcare expenses. This project implements a Multi-Layer Perceptron (MLP) Neural Network to perform regression analysis on the `insurance.csv` dataset, predicting the medical `charges` accurately based on key attributes.

## 📊 Dataset Structure
The project uses the `insurance.csv` dataset, which includes 1,338 records with the following features:
* **age**: Age of the primary beneficiary.
* **sex**: Insurance contractor gender (female, male).
* **bmi**: Body Mass Index, providing an understanding of body weight relative to height.
* **children**: Number of children covered by health insurance / Number of dependents.
* **smoker**: Smoking status (yes, no).
* **region**: The beneficiary's residential area in the US (northeast, northwest, southeast, southwest).
* **charges**: (Target variable) Individual medical costs billed by health insurance.

## 🛠️ Tech Stack & Libraries
* **Python 3**
* **Pandas** & **NumPy** (Data manipulation and analysis)
* **Scikit-Learn** (Data preprocessing and splitting)
* **TensorFlow / Keras** (Building and training the deep learning architecture)

## 🔄 Workflow & Pipeline

### 1. Data Preprocessing
* Loaded the dataset using `pandas`.
* Encoded categorical features (`sex`, `smoker`, `region`) using Scikit-Learn's `LabelEncoder`.
* Split the dataset into features (`X`) and target (`y`).
* Divided data into **80% training** and **20% testing** sets.
* Scaled numerical features using `StandardScaler` to optimize gradient descent.

### 2. Model Architecture
A sequential neural network model was constructed with the following layers:
* **Input Layer**: Derived from the shape of processed features.
* **Hidden Layer 1**: 64 Units with `ReLU` activation function.
* **Hidden Layer 2**: 32 Units with `ReLU` activation function.
* **Output Layer**: 1 Unit (Linear activation) for regression prediction.

### 3. Compilation & Training
* **Optimizer**: Adam
* **Loss Function**: Mean Squared Error (MSE)
* **Evaluation Metrics**: Mean Absolute Error (MAE)
* **Training Settings**: Trained for `100` epochs with a batch size of `32` and a `20%` validation split.

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/insurance-premium-prediction.git](https://github.com/shamkhanayyar18/insurance-premium-prediction.git)
   cd insurance-premium-prediction 
