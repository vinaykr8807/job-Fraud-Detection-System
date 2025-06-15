"use client"

interface SimplePieChartProps {
  data: Array<{ name: string; value: number; fill: string }>
}

export function SimplePieChart({ data }: SimplePieChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-[200px] flex items-center justify-center">
        <p className="text-gray-500">No data available</p>
      </div>
    )
  }

  const total = data.reduce((sum, item) => sum + item.value, 0)
  let currentAngle = 0

  return (
    <div className="w-full h-[200px] flex items-center justify-center">
      <div className="relative">
        <svg width="160" height="160" className="transform -rotate-90">
          {data.map((item, index) => {
            const percentage = (item.value / total) * 100
            const angle = (item.value / total) * 360
            const radius = 70
            const x1 = 80 + radius * Math.cos((currentAngle * Math.PI) / 180)
            const y1 = 80 + radius * Math.sin((currentAngle * Math.PI) / 180)
            const x2 = 80 + radius * Math.cos(((currentAngle + angle) * Math.PI) / 180)
            const y2 = 80 + radius * Math.sin(((currentAngle + angle) * Math.PI) / 180)

            const largeArcFlag = angle > 180 ? 1 : 0
            const pathData = `M 80 80 L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2} Z`

            currentAngle += angle

            return <path key={index} d={pathData} fill={item.fill} stroke="white" strokeWidth="2" />
          })}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-sm font-semibold">{total}</div>
          <div className="text-xs text-gray-500">Total</div>
        </div>
      </div>
      <div className="ml-4 space-y-2">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: item.fill }} />
            <span className="text-sm">
              {item.name}: {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
