"use client"

import { useEffect, useState } from "react"

interface WorkingHistogramProps {
  data: Array<{ range: string; count: number }>
  height?: number
}

export function WorkingHistogram({ data, height = 300 }: WorkingHistogramProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data || data.length === 0) {
    return (
      <div className="w-full flex items-center justify-center" style={{ height: `${height}px` }}>
        <p className="text-gray-500">Loading histogram...</p>
      </div>
    )
  }

  const maxValue = Math.max(...data.map((d) => d.count))

  const getBarColor = (range: string) => {
    const rangeStart = Number.parseInt(range.split("-")[0])
    if (rangeStart < 30) return "#10b981" // Green for low risk
    if (rangeStart < 70) return "#f59e0b" // Orange for medium risk
    return "#ef4444" // Red for high risk
  }

  return (
    <div className="w-full p-4" style={{ height: `${height}px` }}>
      <div className="flex items-end justify-between h-full space-x-1">
        {data.map((item, index) => (
          <div key={index} className="flex flex-col items-center flex-1 h-full">
            <div className="text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">
              {item.count > 0 ? item.count : ""}
            </div>
            <div
              className="w-full rounded-t transition-all duration-300 ease-in-out hover:opacity-80"
              style={{
                backgroundColor: getBarColor(item.range),
                height: `${Math.max((item.count / maxValue) * 80, item.count > 0 ? 2 : 0)}%`,
                minHeight: item.count > 0 ? "2px" : "0px",
              }}
              title={`${item.range}: ${item.count} jobs`}
            />
            <div className="text-xs mt-2 text-center text-gray-500 dark:text-gray-400 transform -rotate-45 origin-center">
              {item.range}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
