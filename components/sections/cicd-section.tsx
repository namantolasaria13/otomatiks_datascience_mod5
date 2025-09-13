"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Workflow, GitBranch, CheckCircle, Zap, Rocket } from "lucide-react"
import { CodeEditor } from "../code-editor"

export function CICDSection() {
  const githubActionsCode = `# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest flake8
    
    - name: Lint code
      run: flake8 src/
    
    - name: Run tests
      run: pytest tests/
    
    - name: Setup DVC
      uses: iterative/setup-dvc@v1
    
    - name: Pull data
      run: dvc pull
      env:
        AWS_ACCESS_KEY_ID: \${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
    - name: Run DVC pipeline
      run: dvc repro
    
    - name: Check metrics
      run: |
        python scripts/check_metrics.py
        dvc metrics show

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment commands here`

  const dockerfileCode = `# Dockerfile for ML service
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`

  const testingCode = `# tests/test_model.py
import pytest
import pandas as pd
import numpy as np
from src.models.predictor import HousePricePredictor

class TestHousePricePredictor:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'bedrooms': [3, 4, 2],
            'bathrooms': [2, 3, 1],
            'sqft_living': [1500, 2000, 1000],
            'price': [300000, 400000, 200000]
        })
    
    @pytest.fixture
    def predictor(self):
        return HousePricePredictor()
    
    def test_model_training(self, predictor, sample_data):
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        
        predictor.train(X, y)
        assert predictor.model is not None
    
    def test_model_prediction(self, predictor, sample_data):
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        
        predictor.train(X, y)
        predictions = predictor.predict(X)
        
        assert len(predictions) == len(X)
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_model_performance(self, predictor, sample_data):
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        
        predictor.train(X, y)
        score = predictor.evaluate(X, y)
        
        assert 0 <= score <= 1  # R² score should be between 0 and 1`

  const metricsCheckCode = `# scripts/check_metrics.py
import json
import sys

def check_model_performance():
    """Check if model meets minimum performance requirements"""
    
    # Load metrics
    with open('metrics/eval_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Define thresholds
    min_accuracy = 0.85
    max_mse = 1000
    
    # Check metrics
    if metrics.get('accuracy', 0) < min_accuracy:
        print(f"❌ Accuracy {metrics['accuracy']:.3f} below threshold {min_accuracy}")
        sys.exit(1)
    
    if metrics.get('mse', float('inf')) > max_mse:
        print(f"❌ MSE {metrics['mse']:.3f} above threshold {max_mse}")
        sys.exit(1)
    
    print("✅ All metrics pass quality gates")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")

if __name__ == "__main__":
    check_model_performance()`

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3 mb-6">
        <Workflow className="h-8 w-8 text-purple-600" />
        <h1 className="text-4xl font-bold text-gray-900">CI/CD for MLOps</h1>
      </div>

      <div className="prose prose-lg max-w-none">
        <p className="text-xl text-gray-700 leading-relaxed">
          Continuous Integration and Continuous Deployment (CI/CD) for machine learning automates the process of
          testing, validating, and deploying ML models. This ensures reliable and reproducible model deployments.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-blue-600" />
              Testing
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">Automated testing of code, data, and model performance</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-green-600" />
              Version Control
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">Trigger pipelines on code changes and pull requests</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Zap className="h-5 w-5 text-purple-600" />
              Automation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">Automated model training and deployment pipelines</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Rocket className="h-5 w-5 text-orange-600" />
              Deployment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">Automated deployment to staging and production</p>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. GitHub Actions Workflow</h3>
          <p className="text-gray-600 mb-4">Set up automated CI/CD pipeline with GitHub Actions:</p>
          <CodeEditor code={githubActionsCode} language="yaml" title="GitHub Actions Workflow" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Docker Configuration</h3>
          <p className="text-gray-600 mb-4">Containerize your ML application for consistent deployments:</p>
          <CodeEditor code={dockerfileCode} language="dockerfile" title="Dockerfile" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Model Testing</h3>
          <p className="text-gray-600 mb-4">Write comprehensive tests for your ML models:</p>
          <CodeEditor code={testingCode} language="python" title="Model Tests" editable />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">4. Quality Gates</h3>
          <p className="text-gray-600 mb-4">Implement quality checks to ensure model performance:</p>
          <CodeEditor code={metricsCheckCode} language="python" title="Metrics Validation" />
        </div>

        <div className="bg-purple-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-900 mb-3">🔄 CI/CD Best Practices</h3>
          <ul className="space-y-2 text-purple-800">
            <li>• Automate all testing (unit, integration, model validation)</li>
            <li>• Use feature branches and pull request reviews</li>
            <li>• Implement quality gates for model performance</li>
            <li>• Separate staging and production environments</li>
            <li>• Monitor deployments and enable rollbacks</li>
            <li>• Use infrastructure as code (Terraform, CloudFormation)</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
