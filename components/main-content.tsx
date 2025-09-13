"use client"

import { IntroSection } from "@/components/sections/intro-section"
import { MLflowSection } from "@/components/sections/mlflow-section"
import { VersioningSection } from "@/components/sections/versioning-section"
import { CICDSection } from "@/components/sections/cicd-section"
import { MonitoringSection } from "@/components/sections/monitoring-section"
import { InfraSection } from "@/components/sections/infra-section"
import { Project1Section } from "@/components/sections/project1-section"
import { Project2Section } from "@/components/sections/project2-section"
import { DemoSection } from "@/components/sections/demo-section"
import { AssignmentSection } from "@/components/sections/assignment-section"

interface MainContentProps {
  activeSection: string
}

export function MainContent({ activeSection }: MainContentProps) {
  const renderSection = () => {
    switch (activeSection) {
      case "intro":
        return <IntroSection />
      case "mlflow":
        return <MLflowSection />
      case "versioning":
        return <VersioningSection />
      case "cicd":
        return <CICDSection />
      case "monitoring":
        return <MonitoringSection />
      case "infra":
        return <InfraSection />
      case "project1":
        return <Project1Section />
      case "project2":
        return <Project2Section />
      case "demo":
        return <DemoSection />
      case "assignment":
        return <AssignmentSection />
      default:
        return <IntroSection />
    }
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="p-8">{renderSection()}</div>
    </div>
  )
}
