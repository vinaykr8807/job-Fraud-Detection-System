"use client"

import { useEffect, useState } from "react"

interface WorkingBarChartProps {
  data: Array<{ category: string; count: number }>
  color?: string
  height?: number
}

export function WorkingBarChart({ data, color = "#3b82f6", height = 200 }: WorkingBarChartProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data || data.length === 0) {
    return (
      <div className="w-full flex items-center justify-center" style={{ height: `${height}px` }}>
        <p className="text-gray-500">Loading chart...</p>
      </div>
    )
  }

  const maxValue = Math.max(...data.map((d) => d.count))

  return (
    <div className="w-full p-4" style={{ height: `${height}px` }}>
      <div className="flex items-end justify-around h-full space-x-4">
        {data.map((item, index) => (
          <div key={index} className="flex flex-col items-center flex-1 h-full">
            <div className="text-sm font-bold mb-2 text-gray-700 dark:text-gray-300">{item.count.toLocaleString()}</div>
            <div
              className="w-full rounded-t-md transition-all duration-500 ease-in-out"
              style={{
                backgroundColor: color,
                height: `${Math.max((item.count / maxValue) * 70, 5)}%`,
                minHeight: "8px",
              }}
            />
            <div className="text-xs mt-3 text-center font-medium text-gray-600 dark:text-gray-400 break-words">
              {item.category}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
