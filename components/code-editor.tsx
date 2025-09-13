"use client"

import { useState } from "react"
import { Copy, Play } from "lucide-react"

interface CodeEditorProps {
  code: string
  language: string
  title?: string
  editable?: boolean
}

export function CodeEditor({ code, language, title, editable = false }: CodeEditorProps) {
  const [currentCode, setCurrentCode] = useState(code)
  const [output, setOutput] = useState("")

  const copyToClipboard = () => {
    navigator.clipboard.writeText(currentCode)
  }

  const runCode = () => {
    setOutput("Code execution simulated - check console for results")
  }

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      {title && <div className="bg-gray-800 px-4 py-2 text-white text-sm font-medium">{title}</div>}
      <div className="relative">
        <div className="absolute top-2 right-2 flex gap-2">
          <button onClick={copyToClipboard} className="p-1 bg-gray-700 hover:bg-gray-600 rounded text-white">
            <Copy className="w-4 h-4" />
          </button>
          {editable && (
            <button onClick={runCode} className="p-1 bg-green-700 hover:bg-green-600 rounded text-white">
              <Play className="w-4 h-4" />
            </button>
          )}
        </div>
        {editable ? (
          <textarea
            value={currentCode}
            onChange={(e) => setCurrentCode(e.target.value)}
            className="w-full p-4 bg-gray-900 text-green-400 font-mono text-sm resize-none"
            rows={code.split("\n").length}
          />
        ) : (
          <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
            <code>{currentCode}</code>
          </pre>
        )}
      </div>
      {output && <div className="bg-gray-800 p-4 text-yellow-400 font-mono text-sm">Output: {output}</div>}
    </div>
  )
}
