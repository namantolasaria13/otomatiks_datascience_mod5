"use client"

import { CodeEditor } from "@/components/code-editor"
import { Database, BarChart3, Package } from "lucide-react"

export function MLflowSection() {
  const setupCode = `# Install MLflow
pip install mlflow

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000`

  const trackingCode = `import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    n_estimators = 100
    max_depth = 6
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Log metrics
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", np.sqrt(mse))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"MSE: {mse}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")`

  const modelRegistryCode = `import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_name = "house-price-predictor"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

# Transition model to staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# Load model for inference
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/Staging"
)

# Make predictions
predictions = model.predict(new_data)`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">MLflow: ML Lifecycle Management</h1>
        <p className="text-lg text-gray-600">
          Learn how to track experiments, manage models, and deploy ML solutions with MLflow
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <BarChart3 className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Experiment Tracking</h3>
          <p className="text-gray-600">Track parameters, metrics, and artifacts across ML experiments</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Package className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Model Registry</h3>
          <p className="text-gray-600">Centralized model store with versioning and stage transitions</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Database className="w-8 h-8 text-purple-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Model Serving</h3>
          <p className="text-gray-600">Deploy models as REST APIs or batch inference jobs</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. MLflow Setup</h3>
          <p className="text-gray-600 mb-4">First, let's install MLflow and start the tracking server:</p>
          <CodeEditor code={setupCode} language="bash" title="Setup MLflow" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Experiment Tracking</h3>
          <p className="text-gray-600 mb-4">Track your ML experiments with parameters, metrics, and models:</p>
          <CodeEditor code={trackingCode} language="python" title="MLflow Tracking" editable />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Model Registry</h3>
          <p className="text-gray-600 mb-4">Register and manage model versions with the MLflow Model Registry:</p>
          <CodeEditor code={modelRegistryCode} language="python" title="Model Registry" />
        </div>

        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Best Practices</h3>
          <ul className="space-y-2 text-blue-800">
            <li>• Use meaningful experiment and run names</li>
            <li>• Log all relevant parameters and hyperparameters</li>
            <li>• Track both training and validation metrics</li>
            <li>• Save model artifacts and preprocessing steps</li>
            <li>• Use tags to organize and filter experiments</li>
            <li>• Set up model staging workflow (Dev → Staging → Production)</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
