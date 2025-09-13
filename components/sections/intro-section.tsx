"use client"

import { ArrowRight, Target, Users, Zap } from "lucide-react"

export function IntroSection() {
  return (
    <div className="max-w-4xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Welcome to MLOps Learning Platform</h1>
        <p className="text-xl text-gray-600">
          Master the art of Machine Learning Operations through hands-on experience
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Target className="w-8 h-8 text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Practical Learning</h3>
          <p className="text-gray-600">Learn by building real MLOps pipelines with industry-standard tools</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Users className="w-8 h-8 text-green-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Interactive Experience</h3>
          <p className="text-gray-600">Engage with live demos, code examples, and hands-on projects</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md">
          <Zap className="w-8 h-8 text-purple-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Production Ready</h3>
          <p className="text-gray-600">Build skills that translate directly to production environments</p>
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8 rounded-lg">
        <h2 className="text-2xl font-bold mb-4">What You'll Learn</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>MLflow for experiment tracking and model management</span>
          </div>
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>DVC for data versioning and pipeline orchestration</span>
          </div>
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>FastAPI for building production ML APIs</span>
          </div>
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>Docker and Kubernetes for deployment</span>
          </div>
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>CI/CD pipelines for automated ML workflows</span>
          </div>
          <div className="flex items-center">
            <ArrowRight className="w-5 h-5 mr-2" />
            <span>Monitoring and observability for ML systems</span>
          </div>
        </div>
      </div>

      <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-xl font-semibold mb-4">Learning Path</h3>
        <div className="space-y-3">
          <div className="flex items-center p-3 bg-blue-50 rounded">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
              1
            </div>
            <span>Start with MLflow basics and experiment tracking</span>
          </div>
          <div className="flex items-center p-3 bg-green-50 rounded">
            <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
              2
            </div>
            <span>Learn data versioning with DVC</span>
          </div>
          <div className="flex items-center p-3 bg-purple-50 rounded">
            <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
              3
            </div>
            <span>Build CI/CD pipelines for ML workflows</span>
          </div>
          <div className="flex items-center p-3 bg-orange-50 rounded">
            <div className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
              4
            </div>
            <span>Implement monitoring and infrastructure</span>
          </div>
          <div className="flex items-center p-3 bg-red-50 rounded">
            <div className="w-8 h-8 bg-red-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
              5
            </div>
            <span>Complete hands-on projects and assignments</span>
          </div>
        </div>
      </div>
    </div>
  )
}
