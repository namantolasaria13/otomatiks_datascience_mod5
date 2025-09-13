"use client"

import { CodeEditor } from "@/components/code-editor"
import { Users, AlertTriangle, BarChart3 } from "lucide-react"

export function Project2Section() {
  const churnModelCode = `# src/models/churn_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import pickle
import yaml

def train_churn_model():
    """Train customer churn prediction model with SMOTE"""
    
    # Load data
    df = pd.read_csv('data/raw/customer_data.csv')
    
    # Feature engineering
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    with mlflow.start_run():
        # Log dataset info
        mlflow.log_param("original_samples", len(X_train))
        mlflow.log_param("balanced_samples", len(X_train_balanced))
        mlflow.log_param("churn_rate", y.mean())
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("auc_score", auc_score)
        mlflow.sklearn.log_model(model, "churn_model")
        
        # Save model
        with open('models/churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model trained! AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_churn_model()`

  const driftDetectionCode = `# src/monitoring/drift_detection.py
import pandas as pd
import numpy as np
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import json

class DriftDetector:
    def __init__(self, reference_data_path):
        self.reference_data = pd.read_csv(reference_data_path)
        self.column_mapping = ColumnMapping(
            target='churn',
            numerical_features=['age', 'tenure', 'monthly_charges', 'total_charges'],
            categorical_features=['gender', 'partner', 'dependents', 'phone_service']
        )
    
    def detect_data_drift(self, current_data):
        """Detect data drift using Evidently AI"""
        
        # Create drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract drift results
        report_dict = data_drift_report.as_dict()
        
        # Calculate overall drift score
        drift_score = self._calculate_drift_score(report_dict)
        
        return {
            'drift_detected': drift_score > 0.1,
            'drift_score': drift_score,
            'detailed_report': report_dict
        }
    
    def detect_target_drift(self, current_data):
        """Detect target drift"""
        
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        return target_drift_report.as_dict()
    
    def _calculate_drift_score(self, report_dict):
        """Calculate overall drift score from report"""
        try:
            metrics = report_dict['metrics']
            drift_scores = []
            
            for metric in metrics:
                if 'result' in metric and 'drift_score' in metric['result']:
                    drift_scores.append(metric['result']['drift_score'])
            
            return np.mean(drift_scores) if drift_scores else 0.0
        except:
            return 0.0
    
    def statistical_drift_test(self, current_data, feature_name):
        """Perform statistical test for drift detection"""
        
        ref_feature = self.reference_data[feature_name]
        curr_feature = current_data[feature_name]
        
        if ref_feature.dtype in ['int64', 'float64']:
            # KS test for numerical features
            statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
        else:
            # Chi-square test for categorical features
            ref_counts = ref_feature.value_counts()
            curr_counts = curr_feature.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
        
        return {
            'feature': feature_name,
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }`

  const streamlitDashboard = `# src/dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import requests

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_model():
    with open('models/churn_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_sample_data():
    return pd.read_csv('data/sample_customers.csv')

def main():
    st.title("🔮 Customer Churn Prediction Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Prediction", "Model Performance", "Data Drift", "System Health"
    ])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "Data Drift":
        drift_page()
    elif page == "System Health":
        health_page()

def prediction_page():
    st.header("Customer Churn Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        # Input fields
        age = st.slider("Age", 18, 80, 35)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
        total_charges = st.slider("Total Charges ($)", 20, 8000, 1500)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        if st.button("Predict Churn Risk"):
            # Prepare features
            features = {
                'age': age,
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'gender': 1 if gender == "Male" else 0,
                'partner': 1 if partner == "Yes" else 0,
                'dependents': 1 if dependents == "Yes" else 0,
                'phone_service': 1 if phone_service == "Yes" else 0
            }
            
            # Make prediction
            model = load_model()
            X = pd.DataFrame([features])
            
            churn_prob = model.predict_proba(X)[0][1]
            churn_pred = model.predict(X)[0]
            
            with col2:
                st.subheader("Prediction Results")
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = churn_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level
                if churn_prob < 0.3:
                    st.success("✅ Low Risk Customer")
                elif churn_prob < 0.7:
                    st.warning("⚠️ Medium Risk Customer")
                else:
                    st.error("🚨 High Risk Customer")
                
                # Recommendations
                st.subheader("Recommendations")
                if churn_prob > 0.5:
                    st.write("• Offer retention incentives")
                    st.write("• Schedule customer success call")
                    st.write("• Provide personalized offers")
                else:
                    st.write("• Continue regular engagement")
                    st.write("• Monitor satisfaction metrics")

def performance_page():
    st.header("Model Performance Monitoring")
    
    # Generate sample metrics over time
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    accuracy = 0.85 + np.random.normal(0, 0.02, len(dates))
    precision = 0.82 + np.random.normal(0, 0.03, len(dates))
    recall = 0.78 + np.random.normal(0, 0.025, len(dates))
    
    metrics_df = pd.DataFrame({
        'date': dates,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })
    
    # Plot metrics
    fig = px.line(metrics_df, x='date', y=['accuracy', 'precision', 'recall'],
                  title='Model Performance Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    # Current metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy[-1]:.3f}", f"{accuracy[-1] - accuracy[-2]:.3f}")
    
    with col2:
        st.metric("Precision", f"{precision[-1]:.3f}", f"{precision[-1] - precision[-2]:.3f}")
    
    with col3:
        st.metric("Recall", f"{recall[-1]:.3f}", f"{recall[-1] - recall[-2]:.3f}")

def drift_page():
    st.header("Data Drift Detection")
    
    # Simulate drift detection results
    st.subheader("Feature Drift Analysis")
    
    features = ['age', 'tenure', 'monthly_charges', 'total_charges']
    drift_scores = np.random.uniform(0, 0.3, len(features))
    
    drift_df = pd.DataFrame({
        'feature': features,
        'drift_score': drift_scores,
        'status': ['Normal' if score < 0.1 else 'Drift Detected' for score in drift_scores]
    })
    
    fig = px.bar(drift_df, x='feature', y='drift_score', color='status',
                 title='Feature Drift Scores')
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(drift_df)

def health_page():
    st.header("System Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "🟢 Healthy", "99.9% uptime")
    
    with col2:
        st.metric("Model Latency", "45ms", "-2ms")
    
    with col3:
        st.metric("Predictions/Hour", "1,247", "+156")
    
    with col4:
        st.metric("Error Rate", "0.1%", "-0.05%")
    
    # System metrics over time
    hours = pd.date_range(start='2024-01-01', periods=24, freq='H')
    latency = 40 + np.random.normal(0, 5, len(hours))
    throughput = 1000 + np.random.normal(0, 100, len(hours))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(x=hours, y=latency, title='API Latency (ms)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(x=hours, y=throughput, title='Throughput (requests/hour)')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()`

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Project 2: Advanced Churn Prediction with Monitoring</h1>
        <p className="text-lg text-gray-600">
          Build a comprehensive churn prediction system with drift detection and monitoring
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Users className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Churn Prediction</h3>
          <p className="text-gray-600">Advanced classification with SMOTE for imbalanced data</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <AlertTriangle className="w-8 h-8 text-orange-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Drift Detection</h3>
          <p className="text-gray-600">Real-time monitoring with Evidently AI</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <BarChart3 className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Interactive Dashboard</h3>
          <p className="text-gray-600">Streamlit dashboard for monitoring and predictions</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">1. Advanced Churn Model</h3>
          <p className="text-gray-600 mb-4">Train a churn prediction model with SMOTE for handling class imbalance:</p>
          <CodeEditor code={churnModelCode} language="python" title="Churn Model Training" editable />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">2. Data Drift Detection</h3>
          <p className="text-gray-600 mb-4">Implement comprehensive drift detection using Evidently AI:</p>
          <CodeEditor code={driftDetectionCode} language="python" title="Drift Detection" />
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">3. Streamlit Monitoring Dashboard</h3>
          <p className="text-gray-600 mb-4">Interactive dashboard for predictions and system monitoring:</p>
          <CodeEditor code={streamlitDashboard} language="python" title="Streamlit Dashboard" />
        </div>

        <div className="bg-purple-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-900 mb-3">🚀 Advanced Features</h3>
          <ul className="space-y-2 text-purple-800">
            <li>• SMOTE for handling imbalanced datasets</li>
            <li>• Real-time data drift detection with Evidently AI</li>
            <li>• Interactive Streamlit dashboard with live monitoring</li>
            <li>• Prometheus metrics integration</li>
            <li>• Automated alerting system</li>
            <li>• A/B testing framework for model comparison</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
