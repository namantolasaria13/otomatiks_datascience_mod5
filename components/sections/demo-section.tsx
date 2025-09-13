"use client"

import { useState } from "react"
import { Play, TrendingUp, AlertTriangle, Activity } from "lucide-react"

export function DemoSection() {
  const [housePrediction, setHousePrediction] = useState<number | null>(null)
  const [churnRisk, setChurnRisk] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const predictHousePrice = async () => {
    setIsLoading(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000))
    const prediction = 450000 + Math.random() * 100000
    setHousePrediction(prediction)
    setIsLoading(false)
  }

  const predictChurnRisk = async () => {
    setIsLoading(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000))
    const risk = Math.random()
    setChurnRisk(risk)
    setIsLoading(false)
  }

  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Live Demo & Interactive Examples</h1>
        <p className="text-lg text-gray-600">Try out the ML models and explore monitoring dashboards</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* House Price Prediction Demo */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center mb-4">
            <TrendingUp className="w-6 h-6 text-blue-600 mr-2" />
            <h3 className="text-xl font-semibold">House Price Prediction</h3>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Bedrooms</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="3" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Bathrooms</label>
                <input type="number" step="0.5" className="w-full p-2 border rounded" defaultValue="2" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sqft Living</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="1500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Year Built</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="1990" />
              </div>
            </div>

            <button
              onClick={predictHousePrice}
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? "Predicting..." : "Predict Price"}
            </button>

            {housePrediction && (
              <div className="bg-blue-50 p-4 rounded">
                <p className="text-blue-900 font-semibold">Predicted Price: ${housePrediction.toLocaleString()}</p>
                <p className="text-blue-700 text-sm mt-1">Confidence: 87%</p>
              </div>
            )}
          </div>
        </div>

        {/* Churn Prediction Demo */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center mb-4">
            <AlertTriangle className="w-6 h-6 text-orange-600 mr-2" />
            <h3 className="text-xl font-semibold">Churn Risk Assessment</h3>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tenure (months)</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="12" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Monthly Charges</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="65" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Contract Type</label>
                <select className="w-full p-2 border rounded">
                  <option>Month-to-month</option>
                  <option>One year</option>
                  <option>Two year</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Support Tickets</label>
                <input type="number" className="w-full p-2 border rounded" defaultValue="2" />
              </div>
            </div>

            <button
              onClick={predictChurnRisk}
              disabled={isLoading}
              className="w-full bg-orange-600 text-white py-2 px-4 rounded hover:bg-orange-700 disabled:opacity-50"
            >
              {isLoading ? "Analyzing..." : "Assess Churn Risk"}
            </button>

            {churnRisk !== null && (
              <div className={`p-4 rounded ${churnRisk > 0.5 ? "bg-red-50" : "bg-green-50"}`}>
                <p className={`font-semibold ${churnRisk > 0.5 ? "text-red-900" : "text-green-900"}`}>
                  Churn Risk: {(churnRisk * 100).toFixed(1)}%
                </p>
                <p className={`text-sm mt-1 ${churnRisk > 0.5 ? "text-red-700" : "text-green-700"}`}>
                  {churnRisk > 0.5 ? "High risk - immediate action recommended" : "Low risk - continue monitoring"}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* System Monitoring Dashboard */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <div className="flex items-center mb-6">
          <Activity className="w-6 h-6 text-green-600 mr-2" />
          <h3 className="text-xl font-semibold">System Monitoring Dashboard</h3>
        </div>

        <div className="grid md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded">
            <p className="text-blue-600 text-sm font-medium">API Requests/min</p>
            <p className="text-2xl font-bold text-blue-900">1,247</p>
            <p className="text-green-600 text-sm">↑ 12% from last hour</p>
          </div>
          <div className="bg-green-50 p-4 rounded">
            <p className="text-green-600 text-sm font-medium">Model Accuracy</p>
            <p className="text-2xl font-bold text-green-900">94.2%</p>
            <p className="text-green-600 text-sm">↑ 0.3% from yesterday</p>
          </div>
          <div className="bg-yellow-50 p-4 rounded">
            <p className="text-yellow-600 text-sm font-medium">Avg Latency</p>
            <p className="text-2xl font-bold text-yellow-900">45ms</p>
            <p className="text-red-600 text-sm">↑ 5ms from baseline</p>
          </div>
          <div className="bg-purple-50 p-4 rounded">
            <p className="text-purple-600 text-sm font-medium">Data Drift Score</p>
            <p className="text-2xl font-bold text-purple-900">0.08</p>
            <p className="text-green-600 text-sm">Within normal range</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-3">Recent Predictions</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                <span className="text-sm">House Price - ID: 1234</span>
                <span className="text-sm font-medium">$485,000</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                <span className="text-sm">Churn Risk - ID: 5678</span>
                <span className="text-sm font-medium text-red-600">High (78%)</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                <span className="text-sm">House Price - ID: 9012</span>
                <span className="text-sm font-medium">$320,000</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-3">System Alerts</h4>
            <div className="space-y-2">
              <div className="flex items-center p-2 bg-yellow-50 rounded">
                <AlertTriangle className="w-4 h-4 text-yellow-600 mr-2" />
                <span className="text-sm">Latency spike detected at 14:30</span>
              </div>
              <div className="flex items-center p-2 bg-green-50 rounded">
                <Activity className="w-4 h-4 text-green-600 mr-2" />
                <span className="text-sm">Model retrained successfully</span>
              </div>
              <div className="flex items-center p-2 bg-blue-50 rounded">
                <TrendingUp className="w-4 h-4 text-blue-600 mr-2" />
                <span className="text-sm">Traffic increased by 15%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-6 rounded-lg">
        <div className="flex items-center mb-4">
          <Play className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-semibold">Try It Yourself</h3>
        </div>
        <p className="mb-4">
          Ready to build your own MLOps pipeline? Follow the step-by-step guides in the previous sections and deploy
          these models in your own environment.
        </p>
        <div className="flex flex-wrap gap-3">
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded text-sm">✅ MLflow Setup</span>
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded text-sm">✅ DVC Pipeline</span>
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded text-sm">✅ FastAPI Service</span>
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded text-sm">✅ Monitoring Dashboard</span>
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded text-sm">✅ CI/CD Pipeline</span>
        </div>
      </div>
    </div>
  )
}
