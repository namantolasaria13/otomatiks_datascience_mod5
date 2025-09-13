"use client"

import { CodeEditor } from "@/components/code-editor"
import { GitBranch, Database, Workflow } from "lucide-react"

export function VersioningSection() {
  const dvcInitCode = `# Initialize DVC in your Git repository
git init
dvc init

# Add remote storage (S3, GCS, Azure, etc.)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Configure remote
dvc remote modify myremote region us-west-2`

  const dataTrackingCode = `# Add data to DVC tracking
dvc add data/raw/dataset.csv
dvc add data/processed/features.pkl

# Commit DVC files to Git
git add data/raw/dataset.csv.dvc data/processed/features.pkl.dvc .dvcignore
git commit -m "Add dataset and features to DVC"

# Push data to remote storage
dvc push

# Push metadata to Git
git push origin main`

  const pipelineCode = `# dvc.yaml - Define your ML pipeline
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
    - src/prepare.py
    - data/raw/dataset.csv
    outs:
    - data/processed/features.pkl
    - data/processed/labels.pkl
    
  train:
    cmd: python src/train.py
    deps:
    - src/train.py
    - data/processed/features.pkl
    - data/processed/labels.pkl
    params:
    - train.learning_rate
    - train.n_estimators
    outs:
    - models/model.pkl
    metrics:
    - metrics/train_metrics.json
    
  evaluate:
    cmd: python src/evaluate.py
    deps:
    - src/evaluate.py
    - models/model.pkl
    - data/processed/features.pkl
    - data/processed/labels.pkl
    outs:
    - metrics/eval_metrics.json`

  const paramsCode = `# params.yaml - Define parameters
train:
  learning_rate: 0.01
  n_estimators: 100
  max_depth: 6
  random_state: 42

prepare:
  test_size: 0.2
  random_state: 42`

  const runPipelineCode = `# Run the entire pipeline
dvc repro

# Run specific stage
dvc repro train

# Show pipeline status
dvc status

# Compare experiments
dvc params diff
dvc metrics diff

# Visualize pipeline
dvc dag`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Data Versioning with DVC</h1>
        <p className="text-lg text-gray-600">Version control for data and ML pipelines - Git for data science</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Database className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Data Versioning</h3>
          <p className="text-gray-600">Track large datasets and model files with Git-like versioning</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Workflow className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Pipeline Management</h3>
          <p className="text-gray-600">Define and reproduce ML pipelines with dependencies</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <GitBranch className="w-8 h-8 text-purple-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Experiment Tracking</h3>
          <p className="text-gray-600">Compare experiments and track parameter changes</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. DVC Setup</h3>
          <p className="text-gray-600 mb-4">Initialize DVC in your Git repository and configure remote storage:</p>
          <CodeEditor code={dvcInitCode} language="bash" title="Initialize DVC" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Data Tracking</h3>
          <p className="text-gray-600 mb-4">Add your datasets and model files to DVC tracking:</p>
          <CodeEditor code={dataTrackingCode} language="bash" title="Track Data with DVC" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Pipeline Definition</h3>
          <p className="text-gray-600 mb-4">Define your ML pipeline stages with dependencies and outputs:</p>
          <CodeEditor code={pipelineCode} language="yaml" title="dvc.yaml" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">4. Parameters Configuration</h3>
          <p className="text-gray-600 mb-4">Define parameters for your pipeline stages:</p>
          <CodeEditor code={paramsCode} language="yaml" title="params.yaml" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">5. Running Pipelines</h3>
          <p className="text-gray-600 mb-4">Execute and manage your ML pipelines:</p>
          <CodeEditor code={runPipelineCode} language="bash" title="Pipeline Execution" />
        </div>

        <div className="bg-green-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-green-900 mb-3">🚀 DVC Benefits</h3>
          <ul className="space-y-2 text-green-800">
            <li>• Version control for large datasets and models</li>
            <li>• Reproducible ML pipelines</li>
            <li>• Efficient storage with deduplication</li>
            <li>• Integration with cloud storage providers</li>
            <li>• Experiment comparison and metrics tracking</li>
            <li>• Collaboration on data science projects</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
