"use client"

import { CodeEditor } from "@/components/code-editor"
import { Home, Database, Zap } from "lucide-react"

export function Project1Section() {
  const projectStructure = `house-price-prediction/
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
│       ├── features.pkl
│       └── labels.pkl
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── monitoring/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml`

  const trainCode = `# src/models/train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import yaml

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['train']

def train_model():
    """Train house price prediction model"""
    
    # Load parameters
    params = load_params()
    
    # Load processed data
    with open('data/processed/features.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/processed/labels.pkl', 'rb') as f:
        y = pickle.load(f)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Log metrics
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Save model
        model_path = "models/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save metrics for DVC
        metrics = {
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2)
        }
        
        with open('metrics/train_metrics.json', 'w') as f:
            import json
            json.dump(metrics, f, indent=2)
        
        print(f"Model trained successfully!")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}")

if __name__ == "__main__":
    train_model()`

  const apiCode = `# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List
import uvicorn

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Load model at startup
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.get("/")
async def root():
    return {"message": "House Price Prediction API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Calculate confidence (simplified)
        confidence = 0.85  # In practice, use prediction intervals
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(features_list: List[HouseFeatures]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        feature_dicts = [f.dict() for f in features_list]
        df = pd.DataFrame(feature_dicts)
        
        # Make predictions
        predictions = model.predict(df)
        
        return {
            "predictions": [float(p) for p in predictions],
            "count": len(predictions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)`

  const dvcPipelineCode = `# dvc.yaml
stages:
  prepare:
    cmd: python src/data/prepare.py
    deps:
    - src/data/prepare.py
    - data/raw/housing.csv
    outs:
    - data/processed/features.pkl
    - data/processed/labels.pkl
    
  train:
    cmd: python src/models/train.py
    deps:
    - src/models/train.py
    - data/processed/features.pkl
    - data/processed/labels.pkl
    params:
    - train.n_estimators
    - train.max_depth
    - train.random_state
    outs:
    - models/model.pkl
    metrics:
    - metrics/train_metrics.json
    
  evaluate:
    cmd: python src/models/evaluate.py
    deps:
    - src/models/evaluate.py
    - models/model.pkl
    - data/processed/features.pkl
    - data/processed/labels.pkl
    metrics:
    - metrics/eval_metrics.json`

  const paramsCode = `# params.yaml
prepare:
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 100
  max_depth: 10
  random_state: 42
  
evaluate:
  threshold: 0.8`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Project 1: House Price Prediction Pipeline</h1>
        <p className="text-lg text-gray-600">
          Build an end-to-end ML pipeline for house price prediction with MLflow, DVC, and FastAPI
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Home className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Regression Model</h3>
          <p className="text-gray-600">Predict house prices using Random Forest regression</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Database className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Data Pipeline</h3>
          <p className="text-gray-600">Automated data processing and feature engineering</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Zap className="w-8 h-8 text-purple-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">API Service</h3>
          <p className="text-gray-600">FastAPI service for real-time predictions</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. Project Structure</h3>
          <p className="text-gray-600 mb-4">Organized project structure following MLOps best practices:</p>
          <CodeEditor code={projectStructure} language="text" title="Project Structure" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Model Training</h3>
          <p className="text-gray-600 mb-4">Train a Random Forest model with MLflow tracking:</p>
          <CodeEditor code={trainCode} language="python" title="Model Training" editable />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. FastAPI Service</h3>
          <p className="text-gray-600 mb-4">Production-ready API for serving predictions:</p>
          <CodeEditor code={apiCode} language="python" title="FastAPI Service" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">4. DVC Pipeline</h3>
          <p className="text-gray-600 mb-4">Define reproducible ML pipeline with DVC:</p>
          <CodeEditor code={dvcPipelineCode} language="yaml" title="DVC Pipeline" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">5. Parameters Configuration</h3>
          <p className="text-gray-600 mb-4">Centralized parameter management:</p>
          <CodeEditor code={paramsCode} language="yaml" title="Parameters" />
        </div>

        <div className="bg-green-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-green-900 mb-3">🎯 Learning Objectives</h3>
          <ul className="space-y-2 text-green-800">
            <li>• Set up MLflow for experiment tracking and model registry</li>
            <li>• Use DVC for data versioning and pipeline orchestration</li>
            <li>• Build production-ready APIs with FastAPI</li>
            <li>• Implement automated testing and CI/CD</li>
            <li>• Deploy with Docker and monitor performance</li>
            <li>• Follow MLOps best practices and project structure</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
