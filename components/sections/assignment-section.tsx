"use client"

import { CheckSquare, Clock, Star, Users } from "lucide-react"

export function AssignmentSection() {
  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Final Assignment: Build Your MLOps Pipeline</h1>
        <p className="text-lg text-gray-600">Apply everything you've learned to build a complete MLOps solution</p>
      </div>

      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8 rounded-lg mb-8">
        <h2 className="text-2xl font-bold mb-4">Assignment Overview</h2>
        <p className="text-lg mb-4">
          Build an end-to-end MLOps pipeline for a machine learning problem of your choice. Your solution should
          demonstrate all the concepts covered in this course.
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="flex items-center">
            <Clock className="w-5 h-5 mr-2" />
            <span>Duration: 2-3 weeks</span>
          </div>
          <div className="flex items-center">
            <Users className="w-5 h-5 mr-2" />
            <span>Individual or Team (2-3)</span>
          </div>
          <div className="flex items-center">
            <Star className="w-5 h-5 mr-2" />
            <span>Weight: 40% of final grade</span>
          </div>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">📋 Requirements</h3>
          <div className="space-y-4">
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">Problem Selection</h4>
                <p className="text-gray-600">
                  Choose a real-world ML problem (classification, regression, or clustering)
                </p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">Data Pipeline</h4>
                <p className="text-gray-600">Implement data versioning with DVC and automated data processing</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">Model Development</h4>
                <p className="text-gray-600">Use MLflow for experiment tracking and model registry</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">API Development</h4>
                <p className="text-gray-600">Create a FastAPI service for model serving</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">CI/CD Pipeline</h4>
                <p className="text-gray-600">Implement automated testing and deployment</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">Monitoring</h4>
                <p className="text-gray-600">Set up model monitoring and drift detection</p>
              </div>
            </div>
            <div className="flex items-start">
              <CheckSquare className="w-5 h-5 text-green-600 mr-3 mt-0.5" />
              <div>
                <h4 className="font-semibold">Deployment</h4>
                <p className="text-gray-600">Deploy using Docker and cloud infrastructure</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">🎯 Suggested Project Ideas</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold text-blue-900">E-commerce Recommendation</h4>
              <p className="text-gray-600 text-sm">
                Build a product recommendation system with collaborative filtering
              </p>
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">Recommendation</span>
                <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">A/B Testing</span>
              </div>
            </div>
            <div className="border-l-4 border-green-500 pl-4">
              <h4 className="font-semibold text-green-900">Fraud Detection</h4>
              <p className="text-gray-600 text-sm">Detect fraudulent transactions with real-time scoring</p>
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">Classification</span>
                <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">Real-time</span>
              </div>
            </div>
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold text-purple-900">Demand Forecasting</h4>
              <p className="text-gray-600 text-sm">Predict product demand using time series analysis</p>
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">Time Series</span>
                <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">Forecasting</span>
              </div>
            </div>
            <div className="border-l-4 border-orange-500 pl-4">
              <h4 className="font-semibold text-orange-900">Image Classification</h4>
              <p className="text-gray-600 text-sm">Build a computer vision model for image recognition</p>
              <div className="mt-2 flex flex-wrap gap-1">
                <span className="bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded">Computer Vision</span>
                <span className="bg-orange-100 text-orange-800 text-xs px-2 py-1 rounded">Deep Learning</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">📊 Evaluation Criteria</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-medium">Technical Implementation (40%)</span>
              <span className="text-sm text-gray-600">Code quality, architecture, best practices</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-medium">MLOps Pipeline (30%)</span>
              <span className="text-sm text-gray-600">CI/CD, monitoring, deployment</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-medium">Documentation (15%)</span>
              <span className="text-sm text-gray-600">README, API docs, setup instructions</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-medium">Presentation (15%)</span>
              <span className="text-sm text-gray-600">Demo, explanation, Q&A</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">📅 Timeline & Deliverables</h3>
          <div className="space-y-4">
            <div className="flex items-center p-3 border-l-4 border-blue-500 bg-blue-50">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                1
              </div>
              <div>
                <h4 className="font-semibold">Week 1: Project Proposal</h4>
                <p className="text-sm text-gray-600">Submit problem statement, dataset, and approach</p>
              </div>
            </div>
            <div className="flex items-center p-3 border-l-4 border-green-500 bg-green-50">
              <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                2
              </div>
              <div>
                <h4 className="font-semibold">Week 2: Implementation</h4>
                <p className="text-sm text-gray-600">Build pipeline, train models, set up monitoring</p>
              </div>
            </div>
            <div className="flex items-center p-3 border-l-4 border-purple-500 bg-purple-50">
              <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                3
              </div>
              <div>
                <h4 className="font-semibold">Week 3: Deployment & Presentation</h4>
                <p className="text-sm text-gray-600">Deploy to cloud, create documentation, present results</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200">
          <h3 className="text-lg font-semibold text-yellow-900 mb-3">💡 Tips for Success</h3>
          <ul className="space-y-2 text-yellow-800">
            <li>• Start with a simple problem and iterate</li>
            <li>• Focus on end-to-end pipeline rather than model complexity</li>
            <li>• Document your decisions and trade-offs</li>
            <li>• Test your pipeline thoroughly</li>
            <li>• Prepare a compelling demo for presentation</li>
            <li>• Use the provided project templates as starting points</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
