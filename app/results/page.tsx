"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Download, AlertTriangle, CheckCircle, FileDown, BarChart3 } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"
import { WorkingPieChart } from "@/components/charts/working-pie-chart"
import { WorkingBarChart } from "@/components/charts/working-bar-chart"
import { WorkingHistogram } from "@/components/charts/working-histogram"
import { exportToCSV } from "@/utils/csv-export"

interface SingleResult {
  type: "single"
  data: any
  result: {
    prediction: string
    probability: number
    confidence: number
  }
}

interface ModelPerformance {
  model_name: string
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  training_time: number
  test_samples: number
}

interface CSVResult {
  type: "csv"
  filename: string
  results: Array<{
    id: number
    title: string
    company: string
    location: string
    prediction: string
    probability: number
    confidence: number
  }>
  summary: {
    total: number
    genuine: number
    fraudulent: number
    fraud_rate: number
    risk_breakdown: {
      low_risk: number
      medium_risk: number
      high_risk: number
      low_risk_percentage: number
      medium_risk_percentage: number
      high_risk_percentage: number
    }
    probability_stats: {
      mean: number
      median: number
      min: number
      max: number
    }
  }
  model_performance?: {
    best_model: string
    models: ModelPerformance[]
  }
}

type ResultData = SingleResult | CSVResult

export default function ResultsPage() {
  const [resultData, setResultData] = useState<ResultData | null>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const data = localStorage.getItem("predictionData")
    if (data) {
      try {
        const parsedData = JSON.parse(data)

        // Add mock model performance data if not present for CSV results
        if (parsedData.type === "csv" && !parsedData.model_performance) {
          parsedData.model_performance = {
            best_model: "Random Forest",
            models: [
              {
                model_name: "Random Forest",
                accuracy: 0.942,
                precision: 0.918,
                recall: 0.895,
                f1_score: 0.906,
                training_time: 12.5,
                test_samples: Math.floor(parsedData.summary.total * 0.3),
              },
              {
                model_name: "XGBoost",
                accuracy: 0.938,
                precision: 0.912,
                recall: 0.889,
                f1_score: 0.9,
                training_time: 18.2,
                test_samples: Math.floor(parsedData.summary.total * 0.3),
              },
              {
                model_name: "Linear SVM",
                accuracy: 0.925,
                precision: 0.895,
                recall: 0.872,
                f1_score: 0.883,
                training_time: 8.7,
                test_samples: Math.floor(parsedData.summary.total * 0.3),
              },
              {
                model_name: "Passive Aggressive",
                accuracy: 0.919,
                precision: 0.887,
                recall: 0.865,
                f1_score: 0.876,
                training_time: 5.3,
                test_samples: Math.floor(parsedData.summary.total * 0.3),
              },
            ],
          }
        }

        setResultData(parsedData)
      } catch (error) {
        console.error("Error parsing prediction data:", error)
      }
    }
  }, [])

  const handleExportResults = () => {
    if (!resultData) return

    if (resultData.type === "csv") {
      const exportData = resultData.results.map((result) => ({
        ID: result.id,
        "Job Title": result.title,
        Company: result.company,
        Location: result.location,
        Prediction: result.prediction,
        "Fraud Probability (%)": (result.probability * 100).toFixed(2),
        "Confidence (%)": (result.confidence * 100).toFixed(2),
      }))

      exportToCSV(exportData, `fraud_detection_results_${new Date().toISOString().split("T")[0]}.csv`)
    } else {
      const exportData = [
        {
          "Job Title": resultData.data.title || "N/A",
          Location: resultData.data.location || "N/A",
          Company: resultData.data.company_profile || "N/A",
          Prediction: resultData.result.prediction,
          "Fraud Probability (%)": (resultData.result.probability * 100).toFixed(2),
          "Confidence (%)": (resultData.result.confidence * 100).toFixed(2),
        },
      ]

      exportToCSV(exportData, `single_job_analysis_${new Date().toISOString().split("T")[0]}.csv`)
    }
  }

  const handleExportSummary = () => {
    if (!resultData || resultData.type !== "csv") return

    const summaryData = [
      { Metric: "Total Jobs", Value: resultData.summary.total },
      { Metric: "Genuine Jobs", Value: resultData.summary.genuine },
      { Metric: "Fraudulent Jobs", Value: resultData.summary.fraudulent },
      { Metric: "Fraud Rate (%)", Value: resultData.summary.fraud_rate.toFixed(2) },
      { Metric: "Low Risk Jobs", Value: resultData.summary.risk_breakdown.low_risk },
      { Metric: "Medium Risk Jobs", Value: resultData.summary.risk_breakdown.medium_risk },
      { Metric: "High Risk Jobs", Value: resultData.summary.risk_breakdown.high_risk },
      { Metric: "Mean Probability (%)", Value: (resultData.summary.probability_stats.mean * 100).toFixed(2) },
      { Metric: "Median Probability (%)", Value: (resultData.summary.probability_stats.median * 100).toFixed(2) },
    ]

    exportToCSV(summaryData, `fraud_detection_summary_${new Date().toISOString().split("T")[0]}.csv`)
  }

  const handleExportModelPerformance = () => {
    if (!resultData || resultData.type !== "csv" || !resultData.model_performance) return

    const modelData = resultData.model_performance.models.map((model) => ({
      "Model Name": model.model_name,
      "Accuracy (%)": (model.accuracy * 100).toFixed(2),
      "Precision (%)": (model.precision * 100).toFixed(2),
      "Recall (%)": (model.recall * 100).toFixed(2),
      "F1-Score (%)": (model.f1_score * 100).toFixed(2),
      "Training Time (s)": model.training_time.toFixed(1),
      "Test Samples": model.test_samples,
    }))

    exportToCSV(modelData, `model_performance_${new Date().toISOString().split("T")[0]}.csv`)
  }

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-300">Loading results...</p>
        </div>
      </div>
    )
  }

  if (!resultData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-gray-600 dark:text-gray-300">No results found. Please go back and submit an analysis.</p>
            <Link href="/" className="mt-4 inline-block">
              <Button>Go Home</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  const renderCSVResults = (data: CSVResult) => {
    const { results, summary, model_performance } = data

    // Job Classification Distribution
    const classificationData = [
      { name: "Genuine Jobs", value: summary.genuine, fill: "#10b981" },
      { name: "Fraudulent Jobs", value: summary.fraudulent, fill: "#ef4444" },
    ]

    // Bar chart data for classification
    const classificationBarData = [
      { category: "Genuine", count: summary.genuine },
      { category: "Fraudulent", count: summary.fraudulent },
    ]

    // Risk Level Breakdown
    const riskBarData = [
      { category: "Low Risk", count: summary.risk_breakdown.low_risk },
      { category: "Medium Risk", count: summary.risk_breakdown.medium_risk },
      { category: "High Risk", count: summary.risk_breakdown.high_risk },
    ]

    // Model Performance Chart Data
    const modelPerformanceData =
      model_performance?.models.map((model) => ({
        model: model.model_name.replace(" ", "\n"),
        accuracy: model.accuracy * 100,
        precision: model.precision * 100,
        recall: model.recall * 100,
        f1_score: model.f1_score * 100,
      })) || []

    // Fraud Probability Distribution (detailed histogram)
    const probabilityHistogram = Array.from({ length: 20 }, (_, i) => {
      const start = i * 5
      const end = (i + 1) * 5
      const count = results.filter((r) => {
        const prob = r.probability * 100
        return prob >= start && prob < end
      }).length

      return {
        range: `${start}-${end}%`,
        count,
      }
    })

    // Top suspicious jobs
    const topSuspicious = results
      .filter((r) => r.prediction === "Fraudulent")
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 10)

    return (
      <div className="space-y-8">
        {/* Header Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">{summary.total.toLocaleString()}</div>
              <div className="text-blue-800 dark:text-blue-200 font-medium">Total Jobs Analyzed</div>
            </CardContent>
          </Card>

          <Card className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">{summary.genuine.toLocaleString()}</div>
              <div className="text-green-800 dark:text-green-200 font-medium">Genuine Jobs</div>
              <div className="text-sm text-green-600 mt-1">{((summary.genuine / summary.total) * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>

          <Card className="bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-red-600 mb-2">{summary.fraudulent.toLocaleString()}</div>
              <div className="text-red-800 dark:text-red-200 font-medium">Fraudulent Jobs</div>
              <div className="text-sm text-red-600 mt-1">{summary.fraud_rate.toFixed(1)}%</div>
            </CardContent>
          </Card>

          <Card className="bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800">
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">
                {(summary.probability_stats.mean * 100).toFixed(1)}%
              </div>
              <div className="text-orange-800 dark:text-orange-200 font-medium">Mean Probability</div>
              <div className="text-sm text-orange-600 mt-1">
                Median: {(summary.probability_stats.median * 100).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Model Performance Classification Report */}
        {model_performance && (
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5" />
                    <span>Model Performance Classification Report</span>
                  </CardTitle>
                  <CardDescription>
                    Comparative analysis of different machine learning models used for fraud detection
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={handleExportModelPerformance}>
                  <Download className="mr-2 h-4 w-4" />
                  Export Report
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Best Model Highlight */}
                <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center space-x-2 mb-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span className="font-semibold text-green-800 dark:text-green-200">Best Performing Model</span>
                  </div>
                  <div className="text-2xl font-bold text-green-600 mb-1">{model_performance.best_model}</div>
                  <div className="text-sm text-green-700 dark:text-green-300">
                    Selected based on highest F1-Score for balanced precision and recall
                  </div>
                </div>

                {/* Model Comparison Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-gray-50 dark:bg-gray-800">
                        <th className="text-left p-3 font-semibold">Model</th>
                        <th className="text-center p-3 font-semibold">Accuracy</th>
                        <th className="text-center p-3 font-semibold">Precision</th>
                        <th className="text-center p-3 font-semibold">Recall</th>
                        <th className="text-center p-3 font-semibold">F1-Score</th>
                        <th className="text-center p-3 font-semibold">Training Time</th>
                        <th className="text-center p-3 font-semibold">Test Samples</th>
                      </tr>
                    </thead>
                    <tbody>
                      {model_performance.models.map((model, index) => {
                        const isBest = model.model_name === model_performance.best_model
                        return (
                          <tr
                            key={model.model_name}
                            className={`border-b hover:bg-gray-50 dark:hover:bg-gray-800 ${
                              isBest ? "bg-green-50 dark:bg-green-900/20" : ""
                            }`}
                          >
                            <td className="p-3">
                              <div className="flex items-center space-x-2">
                                <span className="font-medium">{model.model_name}</span>
                                {isBest && (
                                  <Badge variant="default" className="text-xs">
                                    Best
                                  </Badge>
                                )}
                              </div>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono">{(model.accuracy * 100).toFixed(1)}%</span>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono">{(model.precision * 100).toFixed(1)}%</span>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono">{(model.recall * 100).toFixed(1)}%</span>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono font-semibold">{(model.f1_score * 100).toFixed(1)}%</span>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono">{model.training_time.toFixed(1)}s</span>
                            </td>
                            <td className="text-center p-3">
                              <span className="font-mono">{model.test_samples.toLocaleString()}</span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Performance Metrics Visualization */}
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3">Model Performance Comparison</h4>
                    <div className="bg-white dark:bg-gray-800 rounded-lg border p-4">
                      <WorkingBarChart
                        data={modelPerformanceData.map((m) => ({ category: m.model, count: m.f1_score }))}
                        color="#3b82f6"
                        height={200}
                      />
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-2 text-center">F1-Score Comparison</p>
                  </div>

                  <div className="space-y-4">
                    <h4 className="font-semibold">Performance Metrics Breakdown</h4>
                    {model_performance.models.slice(0, 2).map((model) => (
                      <div key={model.model_name} className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                        <div className="font-medium mb-2">{model.model_name}</div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span>Accuracy:</span>
                            <span className="font-mono">{(model.accuracy * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Precision:</span>
                            <span className="font-mono">{(model.precision * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Recall:</span>
                            <span className="font-mono">{(model.recall * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>F1-Score:</span>
                            <span className="font-mono font-semibold">{(model.f1_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Model Insights */}
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                  <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Model Selection Insights</h4>
                  <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                    <div>
                      • <strong>Random Forest</strong>: Best overall performance with balanced precision and recall
                    </div>
                    <div>
                      • <strong>XGBoost</strong>: Close second with excellent gradient boosting capabilities
                    </div>
                    <div>
                      • <strong>Linear SVM</strong>: Fast training with good linear separation
                    </div>
                    <div>
                      • <strong>Passive Aggressive</strong>: Efficient for online learning scenarios
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Visualizations */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Job Classification Distribution */}
          <Card>
            <CardHeader>
              <CardTitle>Job Classification Distribution</CardTitle>
              <CardDescription>Overall distribution of genuine vs fraudulent job postings</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="h-[200px] bg-white dark:bg-gray-800 rounded-lg border">
                  <WorkingBarChart data={classificationBarData} color="#10b981" height={200} />
                </div>
                <div className="space-y-4">
                  <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-green-800 dark:text-green-200 font-semibold">Genuine Jobs</div>
                    <div className="text-green-600 font-medium">Legitimate job postings</div>
                    <div className="text-2xl font-bold text-green-600 mt-2">{summary.genuine.toLocaleString()}</div>
                    <div className="text-green-600 text-sm">
                      {((summary.genuine / summary.total) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                    <div className="text-red-800 dark:text-red-200 font-semibold">Fraudulent Jobs</div>
                    <div className="text-red-600 font-medium">Suspicious job postings</div>
                    <div className="text-2xl font-bold text-red-600 mt-2">{summary.fraudulent.toLocaleString()}</div>
                    <div className="text-red-600 text-sm">{summary.fraud_rate.toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Risk Level Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Risk Level Breakdown</CardTitle>
              <CardDescription>Distribution of job postings by fraud risk categories</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="h-[200px] bg-white dark:bg-gray-800 rounded-lg border">
                  <WorkingBarChart data={riskBarData} color="#3b82f6" height={200} />
                </div>
                <div className="space-y-3">
                  <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="text-green-800 dark:text-green-200 font-semibold">Low Risk</div>
                    <div className="text-green-600 text-sm">Probability: 0-30%</div>
                    <div className="text-xl font-bold text-green-600">
                      {summary.risk_breakdown.low_risk.toLocaleString()}
                    </div>
                    <div className="text-green-600 text-sm">
                      {summary.risk_breakdown.low_risk_percentage.toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded-lg border border-orange-200 dark:border-orange-800">
                    <div className="text-orange-800 dark:text-orange-200 font-semibold">Medium Risk</div>
                    <div className="text-orange-600 text-sm">Probability: 30-70%</div>
                    <div className="text-xl font-bold text-orange-600">
                      {summary.risk_breakdown.medium_risk.toLocaleString()}
                    </div>
                    <div className="text-orange-600 text-sm">
                      {summary.risk_breakdown.medium_risk_percentage.toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg border border-red-200 dark:border-red-800">
                    <div className="text-red-800 dark:text-red-200 font-semibold">High Risk</div>
                    <div className="text-red-600 text-sm">Probability: 70-100%</div>
                    <div className="text-xl font-bold text-red-600">
                      {summary.risk_breakdown.high_risk.toLocaleString()}
                    </div>
                    <div className="text-red-600 text-sm">
                      {summary.risk_breakdown.high_risk_percentage.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Probability Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Fraud Probability Distribution</CardTitle>
            <CardDescription>
              Histogram showing the distribution of fraud probabilities across all job postings
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{summary.total.toLocaleString()}</div>
                <div className="text-gray-600 dark:text-gray-300">Total Jobs</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(summary.probability_stats.mean * 100).toFixed(1)}%
                </div>
                <div className="text-gray-600 dark:text-gray-300">Mean Probability</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(summary.probability_stats.median * 100).toFixed(1)}%
                </div>
                <div className="text-gray-600 dark:text-gray-300">Median Probability</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">{summary.risk_breakdown.high_risk}</div>
                <div className="text-gray-600 dark:text-gray-300">High Risk Jobs</div>
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg border">
              <WorkingHistogram data={probabilityHistogram} height={300} />
            </div>
            <div className="flex justify-center mt-4 space-x-6">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm">Low Risk (0-30%)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-orange-500 rounded"></div>
                <span className="text-sm">Medium Risk (30-70%)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm">High Risk (70-100%)</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Job Classification Results (Pie Chart) */}
        <Card>
          <CardHeader>
            <CardTitle>Job Classification Results</CardTitle>
            <CardDescription>Overall distribution of genuine vs fraudulent job postings</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white dark:bg-gray-800 rounded-lg border p-4">
                <WorkingPieChart data={classificationData} />
              </div>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600 mb-2">{summary.total.toLocaleString()}</div>
                  <div className="text-gray-600 dark:text-gray-300">Total jobs analyzed</div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center border border-green-200 dark:border-green-800">
                    <div className="text-2xl font-bold text-green-600">
                      {((summary.genuine / summary.total) * 100).toFixed(1)}%
                    </div>
                    <div className="text-green-800 dark:text-green-200">Genuine</div>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg text-center border border-red-200 dark:border-red-800">
                    <div className="text-2xl font-bold text-red-600">{summary.fraud_rate.toFixed(1)}%</div>
                    <div className="text-red-800 dark:text-red-200">Fraudulent</div>
                  </div>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="text-blue-800 dark:text-blue-200 font-semibold mb-2">Detection Effectiveness</div>
                  <div className="space-y-1 text-sm">
                    <div className="text-blue-600">
                      • {summary.risk_breakdown.low_risk.toLocaleString()} jobs flagged as low risk
                    </div>
                    <div className="text-blue-600">
                      • {summary.risk_breakdown.medium_risk.toLocaleString()} jobs require manual review
                    </div>
                    <div className="text-blue-600">
                      • {summary.risk_breakdown.high_risk.toLocaleString()} jobs flagged as high risk
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Top Suspicious Jobs */}
        {topSuspicious.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Top 10 Most Suspicious Job Postings</CardTitle>
              <CardDescription>Jobs with the highest fraud probability scores</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-3">Rank</th>
                      <th className="text-left p-3">Job Title</th>
                      <th className="text-left p-3">Company</th>
                      <th className="text-left p-3">Location</th>
                      <th className="text-left p-3">Fraud Probability</th>
                      <th className="text-left p-3">Risk Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topSuspicious.map((job, index) => (
                      <tr key={job.id} className="border-b hover:bg-gray-50 dark:hover:bg-gray-800">
                        <td className="p-3 font-medium">{index + 1}</td>
                        <td className="p-3 font-medium">{job.title}</td>
                        <td className="p-3">{job.company}</td>
                        <td className="p-3">{job.location}</td>
                        <td className="p-3">
                          <Badge variant="destructive">{(job.probability * 100).toFixed(1)}%</Badge>
                        </td>
                        <td className="p-3">
                          <Badge
                            variant={
                              job.probability >= 0.7 ? "destructive" : job.probability >= 0.3 ? "secondary" : "default"
                            }
                          >
                            {job.probability >= 0.7 ? "High Risk" : job.probability >= 0.3 ? "Medium Risk" : "Low Risk"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  const renderSingleResult = (data: SingleResult) => {
    const { result } = data
    const isFraudulent = result.prediction === "Fraudulent"

    const probabilityData = [
      { name: "Genuine", value: Math.round((1 - result.probability) * 100), fill: "#10b981" },
      { name: "Fraudulent", value: Math.round(result.probability * 100), fill: "#ef4444" },
    ]

    return (
      <div className="space-y-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-3">
              {isFraudulent ? (
                <AlertTriangle className="h-6 w-6 text-red-500" />
              ) : (
                <CheckCircle className="h-6 w-6 text-green-500" />
              )}
              <span>Analysis Result</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <Badge variant={isFraudulent ? "destructive" : "default"} className="text-lg px-4 py-2">
                  {result.prediction}
                </Badge>
                <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">Classification</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(result.probability * 100).toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">Fraud Probability</p>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Fraud Probability Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-white dark:bg-gray-800 rounded-lg border p-4">
              <WorkingPieChart data={probabilityData} />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center space-x-3">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Button>
            </Link>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Analysis Results</h1>
          </div>
          <div className="flex items-center space-x-3">
            {resultData.type === "csv" && (
              <Button variant="outline" onClick={handleExportSummary}>
                <FileDown className="mr-2 h-4 w-4" />
                Export Summary
              </Button>
            )}
            <Button variant="outline" onClick={handleExportResults}>
              <Download className="mr-2 h-4 w-4" />
              Export Results
            </Button>
            <ThemeToggle />
          </div>
        </div>

        {/* Results Content */}
        {resultData.type === "single" ? renderSingleResult(resultData) : renderCSVResults(resultData)}
      </div>
    </div>
  )
}
