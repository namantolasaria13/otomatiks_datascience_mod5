"use client"

import { BookOpen, GitBranch, Zap, BarChart3, Server, CheckSquare, Play, Database } from "lucide-react"

interface SidebarProps {
  activeSection: string
  setActiveSection: (section: string) => void
}

export function Sidebar({ activeSection, setActiveSection }: SidebarProps) {
  const sections = [
    { id: "intro", label: "Introduction", icon: BookOpen },
    { id: "mlflow", label: "MLflow", icon: Database },
    { id: "versioning", label: "Data Versioning", icon: GitBranch },
    { id: "cicd", label: "CI/CD", icon: Zap },
    { id: "monitoring", label: "Monitoring", icon: BarChart3 },
    { id: "infra", label: "Infrastructure", icon: Server },
    { id: "project1", label: "Project 1", icon: CheckSquare },
    { id: "project2", label: "Project 2", icon: CheckSquare },
    { id: "demo", label: "Live Demo", icon: Play },
    { id: "assignment", label: "Assignment", icon: CheckSquare },
  ]

  return (
    <div className="w-64 bg-white shadow-lg">
      <div className="p-6">
        <h1 className="text-xl font-bold text-gray-800">MLOps Learning</h1>
        <p className="text-sm text-gray-600 mt-2">Interactive Platform</p>
      </div>
      <nav className="mt-6">
        {sections.map((section) => {
          const Icon = section.icon
          return (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`w-full flex items-center px-6 py-3 text-left hover:bg-blue-50 transition-colors ${
                activeSection === section.id ? "bg-blue-100 border-r-2 border-blue-500 text-blue-700" : "text-gray-700"
              }`}
            >
              <Icon className="w-5 h-5 mr-3" />
              {section.label}
            </button>
          )
        })}
      </nav>
    </div>
  )
}
