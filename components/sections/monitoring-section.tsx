"use client"

import { CodeEditor } from "@/components/code-editor"
import { BarChart3, AlertTriangle, Activity } from "lucide-react"

export function MonitoringSection() {
  const prometheusCode = `# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'model-metrics'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/model-metrics'
    scrape_interval: 30s`

  const monitoringCode = `# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np

# Define metrics
prediction_counter = Counter('ml_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('ml_model_accuracy', 'Current model accuracy')
data_drift_score = Gauge('ml_data_drift_score', 'Data drift detection score')

class ModelMonitor:
    def __init__(self):
        self.start_metrics_server()
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(8001)
    
    @prediction_latency.time()
    def track_prediction(self, features, prediction):
        """Track prediction metrics"""
        prediction_counter.inc()
        
        # Log prediction for drift detection
        self.log_prediction_data(features, prediction)
    
    def update_model_accuracy(self, accuracy):
        """Update model accuracy metric"""
        model_accuracy.set(accuracy)
    
    def detect_data_drift(self, reference_data, current_data):
        """Detect data drift using statistical tests"""
        from scipy import stats
        
        drift_scores = []
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # Use KS test for numerical features
                statistic, p_value = stats.ks_2samp(
                    reference_data[column], 
                    current_data[column]
                )
                drift_scores.append(statistic)
        
        avg_drift_score = np.mean(drift_scores)
        data_drift_score.set(avg_drift_score)
        
        return avg_drift_score > 0.1  # Threshold for drift detection`

  const alertingCode = `# alerting/rules.yml
groups:
  - name: ml_model_alerts
    rules:
      - alert: HighPredictionLatency
        expr: ml_prediction_duration_seconds > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "Prediction latency is {{ $value }}s"
      
      - alert: LowModelAccuracy
        expr: ml_model_accuracy < 0.8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Current accuracy is {{ $value }}"
      
      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Drift score is {{ $value }}"
      
      - alert: ModelServiceDown
        expr: up{job="ml-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML API service is down"
          description: "The ML API service has been down for more than 1 minute"`

  const loggingCode = `# src/monitoring/logging_config.py
import logging
import json
from datetime import datetime

class MLLogger:
    def __init__(self, service_name="ml-service"):
        self.service_name = service_name
        self.setup_logging()
    
    def setup_logging(self):
        """Configure structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(self.service_name)
    
    def log_prediction(self, request_id, features, prediction, confidence=None):
        """Log prediction with structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event_type": "prediction",
            "request_id": request_id,
            "features": features,
            "prediction": prediction,
            "confidence": confidence
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_model_performance(self, model_version, metrics):
        """Log model performance metrics"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event_type": "model_performance",
            "model_version": model_version,
            "metrics": metrics
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type, error_message, request_id=None):
        """Log errors with context"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "request_id": request_id
        }
        self.logger.error(json.dumps(log_entry))`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">ML Model Monitoring</h1>
        <p className="text-lg text-gray-600">Monitor model performance, detect drift, and ensure system reliability</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <BarChart3 className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Performance Metrics</h3>
          <p className="text-gray-600">Track accuracy, latency, and throughput in real-time</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <AlertTriangle className="w-8 h-8 text-orange-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Data Drift Detection</h3>
          <p className="text-gray-600">Monitor input data distribution changes over time</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Activity className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">System Health</h3>
          <p className="text-gray-600">Monitor service availability and resource usage</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. Prometheus Configuration</h3>
          <p className="text-gray-600 mb-4">Set up Prometheus to collect ML model metrics:</p>
          <CodeEditor code={prometheusCode} language="yaml" title="prometheus.yml" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Model Monitoring</h3>
          <p className="text-gray-600 mb-4">Implement comprehensive model monitoring with metrics collection:</p>
          <CodeEditor code={monitoringCode} language="python" title="Model Monitoring" editable />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Alerting Rules</h3>
          <p className="text-gray-600 mb-4">Configure alerts for model performance and system issues:</p>
          <CodeEditor code={alertingCode} language="yaml" title="Alerting Rules" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">4. Structured Logging</h3>
          <p className="text-gray-600 mb-4">Implement structured logging for better observability:</p>
          <CodeEditor code={loggingCode} language="python" title="Logging Configuration" />
        </div>

        <div className="bg-orange-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-orange-900 mb-3">📊 Monitoring Best Practices</h3>
          <ul className="space-y-2 text-orange-800">
            <li>• Monitor both technical and business metrics</li>
            <li>• Set up automated alerts for critical issues</li>
            <li>• Track data drift and model degradation</li>
            <li>• Use structured logging for better analysis</li>
            <li>• Implement health checks and circuit breakers</li>
            <li>• Create dashboards for stakeholder visibility</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
