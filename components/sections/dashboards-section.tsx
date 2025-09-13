"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { CodeEditor } from "../code-editor"
import { BarChart3, Database, GitBranch, Zap, Activity, Cloud, Play, RefreshCw } from "lucide-react"

export function DashboardsSection() {
  const [predictionInput, setPredictionInput] = useState("")
  const [predictionResult, setPredictionResult] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const mlflowTrainingCode = `# train_model.py - MLflow Integration
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

def train_model():
    # Start MLflow run
    with mlflow.start_run():
        # Load and prepare data
        data = pd.read_csv("data/fraud_data.csv")
        X = data.drop(['is_fraud'], axis=1)
        y = data['is_fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "fraud_detection_model",
            registered_model_name="FraudDetectionModel"
        )
        
        # Save model locally
        joblib.dump(model, "models/fraud_model.pkl")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        return model

if __name__ == "__main__":
    train_model()`

  const dvcConfigCode = `# .dvc/config - DVC Configuration
[core]
    remote = myremote
    
['remote "myremote"']
    url = s3://my-ml-data-bucket/dvc-storage
    
# dvc.yaml - Pipeline Configuration
stages:
  prepare_data:
    cmd: python src/prepare_data.py
    deps:
    - src/prepare_data.py
    - data/raw/transactions.csv
    outs:
    - data/processed/fraud_data.csv
    
  train:
    cmd: python src/train_model.py
    deps:
    - src/train_model.py
    - data/processed/fraud_data.csv
    params:
    - train.n_estimators
    - train.max_depth
    outs:
    - models/fraud_model.pkl
    metrics:
    - metrics/train_metrics.json
    
  evaluate:
    cmd: python src/evaluate_model.py
    deps:
    - src/evaluate_model.py
    - models/fraud_model.pkl
    - data/processed/fraud_data.csv
    metrics:
    - metrics/eval_metrics.json

# params.yaml - Parameters
train:
  n_estimators: 100
  max_depth: 10
  test_size: 0.2
  random_state: 42

# DVC Commands:
# dvc init                    # Initialize DVC
# dvc add data/raw/           # Track data
# dvc push                    # Push data to remote
# dvc repro                   # Reproduce pipeline
# dvc metrics show            # Show metrics`

  const fastApiCode = `# main.py - FastAPI Model Serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="ML model serving with MLflow integration",
    version="1.0.0"
)

# Load model from MLflow
MODEL_NAME = "FraudDetectionModel"
MODEL_VERSION = "latest"

try:
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )
    logger.info(f"Model {MODEL_NAME} loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class TransactionData(BaseModel):
    amount: float
    merchant_category: str
    hour_of_day: int
    day_of_week: int
    user_age: int
    account_age_days: int

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionData]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = pd.DataFrame([transaction.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            model_version=MODEL_VERSION
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare batch data
        input_data = pd.DataFrame([t.dict() for t in request.transactions])
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "transaction_id": i,
                "is_fraud": bool(pred),
                "fraud_probability": float(prob)
            })
        
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

@app.get("/model/info")
async def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "features": [
            "amount", "merchant_category", "hour_of_day",
            "day_of_week", "user_age", "account_age_days"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`

  const githubActionsCode = `name: MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: 3.9
  MLFLOW_TRACKING_URI: \${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: \${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install great-expectations
    
    - name: Run data validation
      run: |
        python scripts/validate_data.py
        great_expectations checkpoint run data_validation
    
    - name: Upload validation results
      uses: actions/upload-artifact@v3
      with:
        name: data-validation-results
        path: validation_results/

  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: \${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Setup DVC
      run: |
        pip install dvc[s3]
        dvc pull
    
    - name: Train model
      run: |
        python src/train_model.py
        dvc repro
    
    - name: Run model tests
      run: |
        python -m pytest tests/test_model.py -v
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          models/
          metrics/

  model-deployment:
    needs: model-training
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-artifacts
    
    - name: Build Docker image
      run: |
        docker build -t fraud-detection-api:\${{ github.sha }} .
        docker tag fraud-detection-api:\${{ github.sha }} fraud-detection-api:latest
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/fraud-api fraud-api=fraud-detection-api:\${{ github.sha }}
        kubectl rollout status deployment/fraud-api
    
    - name: Run integration tests
      run: |
        python tests/test_api_integration.py
    
    - name: Deploy to production
      if: success()
      run: |
        kubectl set image deployment/fraud-api-prod fraud-api=fraud-detection-api:\${{ github.sha }}
        kubectl rollout status deployment/fraud-api-prod

  monitoring:
    needs: model-deployment
    runs-on: ubuntu-latest
    steps:
    - name: Setup monitoring
      run: |
        echo "Setting up Prometheus metrics"
        echo "Configuring Grafana dashboards"
        echo "Setting up alerting rules"
        echo "Monitoring setup completed"`

  const reactUICode = `// components/FraudDetectionUI.tsx
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Shield, AlertTriangle } from 'lucide-react';

interface PredictionResult {
  is_fraud: boolean;
  fraud_probability: number;
  model_version: string;
}

export function FraudDetectionUI() {
  const [formData, setFormData] = useState({
    amount: '',
    merchant_category: '',
    hour_of_day: '',
    day_of_week: '',
    user_age: '',
    account_age_days: ''
  });
  
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          amount: parseFloat(formData.amount),
          merchant_category: formData.merchant_category,
          hour_of_day: parseInt(formData.hour_of_day),
          day_of_week: parseInt(formData.day_of_week),
          user_age: parseInt(formData.user_age),
          account_age_days: parseInt(formData.account_age_days)
        }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Fraud Detection Model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="amount">Transaction Amount</Label>
              <Input
                id="amount"
                type="number"
                value={formData.amount}
                onChange={(e) => handleInputChange('amount', e.target.value)}
                placeholder="100.00"
              />
            </div>
            <div>
              <Label htmlFor="merchant">Merchant Category</Label>
              <Input
                id="merchant"
                value={formData.merchant_category}
                onChange={(e) => handleInputChange('merchant_category', e.target.value)}
                placeholder="grocery"
              />
            </div>
            <div>
              <Label htmlFor="hour">Hour of Day</Label>
              <Input
                id="hour"
                type="number"
                min="0"
                max="23"
                value={formData.hour_of_day}
                onChange={(e) => handleInputChange('hour_of_day', e.target.value)}
                placeholder="14"
              />
            </div>
            <div>
              <Label htmlFor="day">Day of Week</Label>
              <Input
                id="day"
                type="number"
                min="0"
                max="6"
                value={formData.day_of_week}
                onChange={(e) => handleInputChange('day_of_week', e.target.value)}
                placeholder="1"
              />
            </div>
            <div>
              <Label htmlFor="age">User Age</Label>
              <Input
                id="age"
                type="number"
                value={formData.user_age}
                onChange={(e) => handleInputChange('user_age', e.target.value)}
                placeholder="35"
              />
            </div>
            <div>
              <Label htmlFor="account_age">Account Age (days)</Label>
              <Input
                id="account_age"
                type="number"
                value={formData.account_age_days}
                onChange={(e) => handleInputChange('account_age_days', e.target.value)}
                placeholder="365"
              />
            </div>
          </div>
          
          <Button 
            onClick={handlePredict} 
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Detect Fraud'
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {prediction && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {prediction.is_fraud ? (
                <AlertTriangle className="h-5 w-5 text-red-500" />
              ) : (
                <Shield className="h-5 w-5 text-green-500" />
              )}
              Prediction Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Fraud Status:</span>
                <span className={prediction.is_fraud ? 'text-red-600 font-semibold' : 'text-green-600 font-semibold'}>
                  {prediction.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Fraud Probability:</span>
                <span>{(prediction.fraud_probability * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Model Version:</span>
                <span>{prediction.model_version}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}`

  const handlePredict = async () => {
    setIsLoading(true)
    // Simulate API call
    setTimeout(() => {
      const probability = Math.random()
      const isFraud = probability > 0.5
      setPredictionResult(
        isFraud
          ? `🚨 FRAUD DETECTED (${(probability * 100).toFixed(1)}% confidence)`
          : `✅ LEGITIMATE (${((1 - probability) * 100).toFixed(1)}% confidence)`,
      )
      setIsLoading(false)
    }, 2000)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Interactive MLOps Dashboard</h1>
        <p className="text-muted-foreground">
          Complete MLOps workflow with MLflow, DVC, FastAPI, and automated CI/CD deployment.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Models Deployed</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">+2 this week</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Versions</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">47</div>
            <p className="text-xs text-muted-foreground">DVC tracked</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Requests</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1.2M</div>
            <p className="text-xs text-muted-foreground">Last 30 days</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Uptime</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">99.9%</div>
            <p className="text-xs text-muted-foreground">SLA maintained</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="mlflow" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="mlflow">MLflow</TabsTrigger>
          <TabsTrigger value="dvc">DVC</TabsTrigger>
          <TabsTrigger value="fastapi">FastAPI</TabsTrigger>
          <TabsTrigger value="cicd">CI/CD</TabsTrigger>
          <TabsTrigger value="demo">Live Demo</TabsTrigger>
        </TabsList>

        <TabsContent value="mlflow" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                MLflow Experiment Tracking
              </CardTitle>
              <CardDescription>Track experiments, log metrics, and manage model lifecycle with MLflow</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Experiment Tracking</Badge>
                <Badge variant="secondary">Model Registry</Badge>
                <Badge variant="secondary">Artifact Storage</Badge>
                <Badge variant="secondary">Model Versioning</Badge>
              </div>

              <CodeEditor language="python" code={mlflowTrainingCode} title="MLflow Training Script" />

              <div className="bg-muted p-4 rounded-lg">
                <h4 className="font-semibold mb-2">MLflow Features:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li>Automatic experiment logging and comparison</li>
                  <li>Model registry with versioning and staging</li>
                  <li>Artifact storage for models and datasets</li>
                  <li>Model serving capabilities with REST API</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dvc" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5" />
                DVC Data Version Control
              </CardTitle>
              <CardDescription>Version control for data and ML pipelines with Git integration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Data Versioning</Badge>
                <Badge variant="secondary">Pipeline Automation</Badge>
                <Badge variant="secondary">Remote Storage</Badge>
                <Badge variant="secondary">Git Integration</Badge>
              </div>

              <CodeEditor language="yaml" code={dvcConfigCode} title="DVC Configuration" />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">DVC Benefits:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Version large datasets efficiently</li>
                    <li>Reproducible ML pipelines</li>
                    <li>Remote storage integration</li>
                    <li>Git-like workflow for data</li>
                  </ul>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Pipeline Stages:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Data preparation and cleaning</li>
                    <li>Feature engineering</li>
                    <li>Model training and validation</li>
                    <li>Model evaluation and metrics</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="fastapi" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                FastAPI Model Serving
              </CardTitle>
              <CardDescription>High-performance API for serving ML models with automatic documentation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">FastAPI</Badge>
                <Badge variant="secondary">Pydantic</Badge>
                <Badge variant="secondary">Auto Documentation</Badge>
                <Badge variant="secondary">Async Support</Badge>
              </div>

              <CodeEditor language="python" code={fastApiCode} title="FastAPI Model Server" />

              <div className="bg-muted p-4 rounded-lg">
                <h4 className="font-semibold mb-2">API Endpoints:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li>
                    <code>/health</code> - Health check and model status
                  </li>
                  <li>
                    <code>/predict</code> - Single prediction endpoint
                  </li>
                  <li>
                    <code>/predict/batch</code> - Batch prediction endpoint
                  </li>
                  <li>
                    <code>/model/info</code> - Model metadata and features
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cicd" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                GitHub Actions CI/CD Pipeline
              </CardTitle>
              <CardDescription>Automated pipeline for data validation, model training, and deployment</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">GitHub Actions</Badge>
                <Badge variant="secondary">Data Validation</Badge>
                <Badge variant="secondary">Model Testing</Badge>
                <Badge variant="secondary">Auto Deployment</Badge>
              </div>

              <CodeEditor language="yaml" code={githubActionsCode} title="GitHub Actions Workflow" />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Pipeline Stages:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Data validation with Great Expectations</li>
                    <li>Model training with DVC pipeline</li>
                    <li>Model testing and validation</li>
                    <li>Containerized deployment</li>
                  </ul>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Deployment Strategy:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Staging environment testing</li>
                    <li>Integration test validation</li>
                    <li>Production deployment</li>
                    <li>Monitoring and alerting</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="demo" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Live Fraud Detection Demo
              </CardTitle>
              <CardDescription>Interactive demo of the deployed fraud detection model</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="amount">Transaction Amount ($)</Label>
                  <Input
                    id="amount"
                    type="number"
                    value={predictionInput}
                    onChange={(e) => setPredictionInput(e.target.value)}
                    placeholder="Enter amount (e.g., 150.00)"
                  />
                </div>
                <div className="flex items-end">
                  <Button onClick={handlePredict} disabled={isLoading || !predictionInput} className="w-full">
                    {isLoading ? (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      "Detect Fraud"
                    )}
                  </Button>
                </div>
              </div>

              {predictionResult && (
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Prediction Result:</h4>
                  <p className="text-lg">{predictionResult}</p>
                </div>
              )}

              <div className="bg-muted p-4 rounded-lg">
                <h4 className="font-semibold mb-2">React Frontend Code:</h4>
                <CodeEditor language="typescript" code={reactUICode} title="React UI Component" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
