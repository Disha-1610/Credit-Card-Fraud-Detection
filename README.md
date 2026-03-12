
## Project Description
Credit card fraud is a significant challenge in modern financial systems, leading to substantial financial losses for banks and payment processors. Detecting fraudulent transactions is difficult because fraud cases are extremely rare compared to legitimate transactions.

This project implements a machine learning–based credit card fraud detection system capable of identifying suspicious transactions using an ensemble of classification models. The system combines Logistic Regression, Decision Tree, and Support Vector Machine (SVM) using a soft voting ensemble, improving detection performance on highly imbalanced datasets.

To enable real-world usability, the trained model is integrated with a FastAPI backend, allowing transactions to be evaluated through a REST API for real-time fraud prediction.

The system is designed with production considerations in mind, including model persistence, API-based inference, and containerization, making it suitable for deployment in real-world financial monitoring systems.



## Machine Learning Approach
Fraud detection is treated as a binary classification problem, where each transaction is classified as either:

Legitimate (0)
Fraudulent (1)

To improve model robustness, this project uses an ensemble learning approach, combining multiple classifiers to leverage their individual strengths.

Models Used
Model	Purpose
Logistic Regression	Baseline probabilistic classifier
Decision Tree	Captures non-linear patterns
Support Vector Machine	Effective for high-dimensional feature spaces
Voting Classifier	Aggregates predictions from all models

The ensemble uses soft voting, which averages predicted probabilities from all models to produce the final prediction



## Features
Machine learning–based fraud detection system

Ensemble learning using multiple classifiers

Handles highly imbalanced financial datasets

Real-time transaction prediction API

Backend built using FastAPI

Automatic API documentation via Swagger UI

Model persistence using joblib

Docker-based containerization for deployment

Modular project structure for scalability


## Tech Stack
-Programming Language

Python

-Data Processing

pandas
numpy

-Machine Learning

scikit-learn

-Backend API

FastAPI
Uvicorn

-Model Serialization

joblib

-Deployment

Docker



## Installation

1.Clone the Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2.Create a Virtual Environment
Using a virtual environment ensures that project dependencies do not conflict with your system Python installation.
Windows

```bash
python -m venv venv
venv\Scripts\activate
```
Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```
3.Install Project Dependencies
Install all required Python packages using the provided requirements file.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4.Download the Dataset

Download the Credit Card Fraud Detection dataset from:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset file in the root project directory:

```bash
credit-card-fraud-detection/
│
├── creditcard.csv
├── train_model.py
├── app.py
```
5.Train the Machine Learning Model

Run the training script to train the ensemble model and save the trained model artifacts.

```bash
python train_model.py
```
This step generates the following files:

```bash
fraud_model.pkl
scaler.pkl
```
These files are used by the backend API to perform predictions.
6.Run the Backend API

Start the FastAPI server using Uvicorn:
```bash
uvicorn app:app --reload
```
The API will be available at:
```bash
http://127.0.0.1:8000
```
7.Access API Documentation

FastAPI automatically generates interactive documentation.

Open in your browser:
```bash
http://127.0.0.1:8000/docs
```
This interface allows you to test the fraud prediction endpoint directly.

## API Usage
The backend exposes a REST API endpoint that accepts transaction data and returns a fraud prediction

Endpoint
```bash
POST /predict
```
Request Example
```bash
{
 "Time": 10000,
 "V1": -1.359807,
 "V2": -0.072781,
 "V3": 2.536346,
 "V4": 1.378155,
 "V5": -0.338321,
 "V6": 0.462388,
 "V7": 0.239599,
 "V8": 0.098698,
 "V9": 0.363787,
 "V10": 0.090794,
 "V11": -0.551600,
 "V12": -0.617801,
 "V13": -0.991390,
 "V14": -0.311169,
 "V15": 1.468177,
 "V16": -0.470401,
 "V17": 0.207971,
 "V18": 0.025791,
 "V19": 0.403993,
 "V20": 0.251412,
 "V21": -0.018307,
 "V22": 0.277838,
 "V23": -0.110474,
 "V24": 0.066928,
 "V25": 0.128539,
 "V26": -0.189115,
 "V27": 0.133558,
 "V28": -0.021053,
 "Amount": 149.62
}
```
Response Example
```bash
{
 "prediction": "Fraud",
 "fraud_probability": 0.92
}
```
## Dataset
This project uses the Credit Card Fraud Detection Dataset, which contains anonymized transaction records from European cardholders.

Dataset characteristics:

284,807 transactions

492 fraudulent transactions

Highly imbalanced dataset

Features transformed using Principal Component Analysis (PCA)
## Docker Deployment

The project can be containerized using Docker to ensure consistent environments across development and production.

Build the Docker image:

```bash
  docker build -t fraud-detection-api .
```
Run the container:
```bash
  docker run -p 8000:8000 fraud-detection-api
```

##  Real World Applications
Fraud detection systems similar to this project are widely used in financial technology platforms.

Banking Systems

Banks analyze transaction patterns in real time to detect suspicious activity.

Online Payment Platforms

Companies such as
PayPal, Stripe, and Visa
use machine learning systems to prevent fraudulent payments.

E-commerce Platforms

Online retailers monitor purchases to identify fraudulent transactions.

Insurance Industry

Fraud detection models help identify suspicious insurance claims.
## Future improvements
Potential improvements for production-scale systems include:

Real-time fraud detection using streaming architectures

Model monitoring and drift detection

Automated model retraining pipelines

Deep learning–based anomaly detection models

Feature store integration

Real-time dashboards for fraud analysts
