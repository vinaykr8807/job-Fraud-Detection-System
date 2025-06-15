"use client"

interface SimpleBarChartProps {
  data: Array<{ category: string; count: number }>
  color?: string
}

export function SimpleBarChart({ data, color = "#3b82f6" }: SimpleBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-[200px] flex items-center justify-center">
        <p className="text-gray-500">No data available</p>
      </div>
    )
  }

  const maxValue = Math.max(...data.map((d) => d.count))

  return (
    <div className="w-full h-[200px] p-4">
      <div className="flex items-end justify-around h-full space-x-2">
        {data.map((item, index) => (
          <div key={index} className="flex flex-col items-center flex-1">
            <div className="text-xs font-medium mb-1">{item.count}</div>
            <div
              className="w-full rounded-t"
              style={{
                backgroundColor: color,
                height: `${(item.count / maxValue) * 100}%`,
                minHeight: "4px",
              }}
            />
            <div className="text-xs mt-2 text-center break-words">{item.category}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
