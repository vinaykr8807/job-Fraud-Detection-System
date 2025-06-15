"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface BarChartData {
  [key: string]: string | number
}

interface CustomBarChartProps {
  data: BarChartData[]
  xKey: string
  yKey: string
  title?: string
  color?: string
}

export function CustomBarChart({ data, xKey, yKey, title, color = "#3b82f6" }: CustomBarChartProps) {
  return (
    <div className="w-full h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} angle={-45} textAnchor="end" height={80} fontSize={12} interval={0} />
          <YAxis />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid #ccc",
              borderRadius: "4px",
            }}
          />
          <Bar dataKey={yKey} fill={color} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
