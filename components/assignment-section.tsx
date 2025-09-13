import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, Clock, AlertCircle } from "lucide-react"

export function AssignmentSection() {
  const assignments = [
    {
      title: "Setup MLflow Tracking Server",
      difficulty: "Beginner",
      estimatedTime: "30 minutes",
      status: "available",
      description:
        "Set up a local MLflow tracking server and log your first experiment with parameters, metrics, and artifacts.",
      tasks: [
        "Install MLflow and start tracking server",
        "Create a simple ML experiment",
        "Log parameters, metrics, and model artifacts",
        "Explore the MLflow UI",
      ],
    },
    {
      title: "Build End-to-End ML Pipeline",
      difficulty: "Intermediate",
      estimatedTime: "2 hours",
      status: "available",
      description: "Create a complete ML pipeline using MLflow, DVC, and FastAPI for a fraud detection system.",
      tasks: [
        "Set up DVC for data versioning",
        "Train and register model in MLflow",
        "Create FastAPI service for model serving",
        "Build React frontend for predictions",
        "Test the complete pipeline",
      ],
    },
    {
      title: "Implement CI/CD Pipeline",
      difficulty: "Advanced",
      estimatedTime: "3 hours",
      status: "available",
      description: "Set up automated CI/CD pipeline with GitHub Actions for model training, testing, and deployment.",
      tasks: [
        "Create GitHub Actions workflow",
        "Implement automated testing",
        "Set up model validation gates",
        "Configure automated deployment",
        "Add monitoring and alerting",
      ],
    },
  ]

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-600" />
      case "in-progress":
        return <Clock className="h-5 w-5 text-blue-600" />
      default:
        return <AlertCircle className="h-5 w-5 text-gray-400" />
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case "beginner":
        return "bg-green-100 text-green-800 border-green-200"
      case "intermediate":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "advanced":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3 mb-6">
        <CheckCircle className="h-8 w-8 text-green-600" />
        <h1 className="text-4xl font-bold text-gray-900">Hands-on Assignments</h1>
      </div>

      <div className="prose prose-lg max-w-none">
        <p className="text-xl text-gray-700 leading-relaxed">
          Practice what you've learned with these hands-on assignments. Each assignment builds upon the previous
          concepts and provides real-world experience with MLOps tools and practices.
        </p>
      </div>

      <div className="grid gap-6">
        {assignments.map((assignment, index) => (
          <Card key={index} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(assignment.status)}
                  <div>
                    <CardTitle className="text-xl">{assignment.title}</CardTitle>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge className={getDifficultyColor(assignment.difficulty)}>{assignment.difficulty}</Badge>
                      <Badge variant="outline">
                        <Clock className="h-3 w-3 mr-1" />
                        {assignment.estimatedTime}
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-gray-700 mb-4">{assignment.description}</p>

              <div>
                <h4 className="font-semibold mb-2">Tasks to Complete:</h4>
                <ul className="space-y-1">
                  {assignment.tasks.map((task, taskIndex) => (
                    <li key={taskIndex} className="flex items-center gap-2 text-sm text-gray-600">
                      <div className="w-2 h-2 bg-blue-400 rounded-full flex-shrink-0" />
                      {task}
                    </li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <CardTitle className="text-blue-800">💡 Getting Started Tips</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Prerequisites:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>Python 3.8+ installed</li>
                <li>Git and GitHub account</li>
                <li>Docker (for advanced assignments)</li>
                <li>Basic ML and Python knowledge</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Resources:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>MLflow documentation</li>
                <li>DVC getting started guide</li>
                <li>FastAPI tutorial</li>
                <li>GitHub Actions documentation</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
