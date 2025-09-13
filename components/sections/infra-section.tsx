"use client"

import { CodeEditor } from "@/components/code-editor"
import { Server, Cloud, Container } from "lucide-react"

export function InfraSection() {
  const dockerComposeCode = `# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URL=postgresql://user:password@postgres:5432/mlops
    depends_on:
      - postgres
      - mlflow
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://user:password@postgres:5432/mlflow
      - DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
    command: >
      bash -c "pip install mlflow psycopg2-binary boto3 &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://user:password@postgres:5432/mlflow
               --default-artifact-root s3://mlflow-artifacts"
    depends_on:
      - postgres

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mlops
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/rules.yml:/etc/prometheus/rules.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  grafana_data:`

  const kubernetesCode = `# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80`

  const terraformCode = `# infrastructure/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
resource "aws_eks_cluster" "mlops_cluster" {
  name     = "mlops-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]
}

# S3 Bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "mlops-mlflow-artifacts-\${random_string.bucket_suffix.result}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# RDS for MLflow backend
resource "aws_db_instance" "mlflow_db" {
  identifier = "mlflow-db"
  
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = "mlflow"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}

# Random string for unique naming
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Infrastructure & Deployment</h1>
        <p className="text-lg text-gray-600">Deploy and scale ML systems with modern infrastructure tools</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Container className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Containerization</h3>
          <p className="text-gray-600">Package ML applications with Docker for consistent deployment</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Server className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Orchestration</h3>
          <p className="text-gray-600">Use Kubernetes for scalable and resilient ML services</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Cloud className="w-8 h-8 text-purple-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Cloud Infrastructure</h3>
          <p className="text-gray-600">Provision cloud resources with Infrastructure as Code</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. Docker Compose Setup</h3>
          <p className="text-gray-600 mb-4">Complete development environment with all ML services:</p>
          <CodeEditor code={dockerComposeCode} language="yaml" title="docker-compose.yml" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Kubernetes Deployment</h3>
          <p className="text-gray-600 mb-4">Production-ready Kubernetes manifests with auto-scaling:</p>
          <CodeEditor code={kubernetesCode} language="yaml" title="Kubernetes Deployment" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Infrastructure as Code</h3>
          <p className="text-gray-600 mb-4">Terraform configuration for AWS infrastructure:</p>
          <CodeEditor code={terraformCode} language="hcl" title="Terraform Configuration" />
        </div>

        <div className="bg-blue-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">🏗️ Infrastructure Best Practices</h3>
          <ul className="space-y-2 text-blue-800">
            <li>• Use Infrastructure as Code for reproducible deployments</li>
            <li>• Implement proper security groups and network policies</li>
            <li>• Set up monitoring and logging from the start</li>
            <li>• Use managed services when possible (RDS, EKS, etc.)</li>
            <li>• Implement backup and disaster recovery strategies</li>
            <li>• Use secrets management for sensitive configuration</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
