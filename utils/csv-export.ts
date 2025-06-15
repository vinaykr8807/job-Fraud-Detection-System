export function exportToCSV(data: any[], filename: string) {
  // Convert data to CSV format
  const csvContent = convertToCSV(data)

  // Create blob and download
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
  const link = document.createElement("a")

  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", filename)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
}

function convertToCSV(data: any[]): string {
  if (!data || data.length === 0) return ""

  // Get headers from the first object
  const headers = Object.keys(data[0])

  // Create CSV header row
  const csvHeaders = headers.join(",")

  // Create CSV data rows
  const csvRows = data.map((row) => {
    return headers
      .map((header) => {
        const value = row[header]
        // Handle values that might contain commas or quotes
        if (typeof value === "string" && (value.includes(",") || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value
      })
      .join(",")
  })

  // Combine headers and rows
  return [csvHeaders, ...csvRows].join("\n")
}

export function exportSummaryToCSV(summary: any, filename: string) {
  const summaryData = [
    { Metric: "Total Jobs", Value: summary.total },
    { Metric: "Genuine Jobs", Value: summary.genuine },
    { Metric: "Fraudulent Jobs", Value: summary.fraudulent },
    { Metric: "Fraud Rate (%)", Value: ((summary.fraudulent / summary.total) * 100).toFixed(2) },
  ]

  exportToCSV(summaryData, filename)
}
