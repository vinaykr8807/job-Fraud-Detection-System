"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileText, BarChart3, Shield } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center space-x-3">
            <Shield className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Job Fraud Detection System</h1>
          </div>
          <ThemeToggle />
        </div>

        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Protect Against Fraudulent Job Postings
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Advanced machine learning system to detect and analyze fraudulent job postings with real-time insights and
            comprehensive reporting.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-green-600" />
                <div>
                  <CardTitle>Single Job Analysis</CardTitle>
                  <CardDescription>Analyze individual job postings by filling out a detailed form</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Enter job details manually and get instant fraud probability analysis with detailed insights and
                recommendations.
              </p>
              <Link href="/predict-form">
                <Button className="w-full">
                  <FileText className="mr-2 h-4 w-4" />
                  Analyze Single Job
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center space-x-3">
                <Upload className="h-8 w-8 text-blue-600" />
                <div>
                  <CardTitle>Batch CSV Analysis</CardTitle>
                  <CardDescription>Upload CSV files for bulk job posting analysis</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Upload CSV files containing multiple job postings and get comprehensive analysis with detailed reporting
                and visualizations.
              </p>
              <Link href="/upload-csv">
                <Button className="w-full" variant="outline">
                  <Upload className="mr-2 h-4 w-4" />
                  Upload CSV File
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Features List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
          <h3 className="text-2xl font-bold text-center mb-8 text-gray-900 dark:text-white">Key Features</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-blue-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Real-time Analysis</h4>
                  <p className="text-gray-600 dark:text-gray-300">Instant fraud detection with probability scores</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-green-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Interactive Dashboard</h4>
                  <p className="text-gray-600 dark:text-gray-300">Comprehensive visualizations and insights</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-purple-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Batch Processing</h4>
                  <p className="text-gray-600 dark:text-gray-300">Analyze multiple job postings simultaneously</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-orange-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Model Retraining</h4>
                  <p className="text-gray-600 dark:text-gray-300">Continuous learning from new data</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-red-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Detailed Reports</h4>
                  <p className="text-gray-600 dark:text-gray-300">Comprehensive analysis and recommendations</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <BarChart3 className="h-6 w-6 text-indigo-600 mt-1" />
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Export Capabilities</h4>
                  <p className="text-gray-600 dark:text-gray-300">Download results and reports</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
