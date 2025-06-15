"use client"

import { useEffect, useState } from "react"

interface WorkingPieChartProps {
  data: Array<{ name: string; value: number; fill: string }>
  size?: number
}

export function WorkingPieChart({ data, size = 160 }: WorkingPieChartProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data || data.length === 0) {
    return (
      <div className="w-full h-[300px] flex items-center justify-center">
        <p className="text-gray-500">Loading chart...</p>
      </div>
    )
  }

  const total = data.reduce((sum, item) => sum + item.value, 0)
  let currentAngle = 0

  const createPath = (centerX: number, centerY: number, radius: number, startAngle: number, endAngle: number) => {
    const start = polarToCartesian(centerX, centerY, radius, endAngle)
    const end = polarToCartesian(centerX, centerY, radius, startAngle)
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1"
    return [
      "M",
      centerX,
      centerY,
      "L",
      start.x,
      start.y,
      "A",
      radius,
      radius,
      0,
      largeArcFlag,
      0,
      end.x,
      end.y,
      "Z",
    ].join(" ")
  }

  const polarToCartesian = (centerX: number, centerY: number, radius: number, angleInDegrees: number) => {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians),
    }
  }

  return (
    <div className="w-full h-[300px] flex items-center justify-center">
      <div className="flex items-center space-x-8">
        <div className="relative">
          <svg width={size} height={size} className="drop-shadow-lg">
            {data.map((item, index) => {
              const percentage = (item.value / total) * 100
              const angle = (item.value / total) * 360
              const radius = size / 2 - 10
              const centerX = size / 2
              const centerY = size / 2

              const path = createPath(centerX, centerY, radius, currentAngle, currentAngle + angle)
              currentAngle += angle

              return (
                <path
                  key={index}
                  d={path}
                  fill={item.fill}
                  stroke="white"
                  strokeWidth="2"
                  className="hover:opacity-80 transition-opacity duration-200"
                />
              )
            })}
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-xl font-bold text-gray-800 dark:text-gray-200">{total.toLocaleString()}</div>
            <div className="text-sm text-gray-500">Total</div>
          </div>
        </div>
        <div className="space-y-3">
          {data.map((item, index) => (
            <div key={index} className="flex items-center space-x-3">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: item.fill }} />
              <div className="text-sm">
                <div className="font-medium text-gray-800 dark:text-gray-200">{item.name}</div>
                <div className="text-gray-500">
                  {item.value.toLocaleString()} ({((item.value / total) * 100).toFixed(1)}%)
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
