"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Upload, FileText, Loader2, Download } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"
import { useRouter } from "next/navigation"
import { useDropzone } from "react-dropzone"

export default function UploadCSVPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(false)
  const [file, setFile] = useState<File | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".csv"],
    },
    multiple: false,
  })

  const handleAnalyze = async () => {
    if (!file) return

    setLoading(true)

    // Simulate realistic CSV analysis
    await new Promise((resolve) => setTimeout(resolve, 4000))

    // Generate comprehensive mock results matching the screenshots
    const totalJobs = 3576
    const fraudulentJobs = 28
    const genuineJobs = totalJobs - fraudulentJobs

    // Generate realistic job data
    const jobTitles = [
      "Software Engineer",
      "Data Scientist",
      "Marketing Manager",
      "Sales Representative",
      "Customer Service Rep",
      "Project Manager",
      "Business Analyst",
      "Web Developer",
      "Graphic Designer",
      "Account Manager",
      "HR Specialist",
      "Financial Analyst",
      "Product Manager",
      "Operations Manager",
      "Content Writer",
      "UX Designer",
    ]

    const companies = [
      "TechCorp Inc",
      "DataSolutions LLC",
      "Marketing Pro",
      "Sales Force Co",
      "Customer First",
      "Project Masters",
      "Analytics Plus",
      "WebDev Studio",
      "Design Hub",
      "Account Pro",
      "HR Solutions",
      "Finance Group",
    ]

    const locations = [
      "New York, NY",
      "San Francisco, CA",
      "Chicago, IL",
      "Austin, TX",
      "Seattle, WA",
      "Boston, MA",
      "Los Angeles, CA",
      "Denver, CO",
      "Miami, FL",
      "Atlanta, GA",
      "Portland, OR",
      "Phoenix, AZ",
    ]

    const mockResults = Array.from({ length: totalJobs }, (_, i) => {
      // Create realistic probability distribution
      let probability
      if (i < fraudulentJobs) {
        // Fraudulent jobs - higher probabilities
        probability = 0.7 + Math.random() * 0.3
      } else {
        // Genuine jobs - lower probabilities with realistic distribution
        const rand = Math.random()
        if (rand < 0.6) {
          probability = Math.random() * 0.1 // 0-10%
        } else if (rand < 0.85) {
          probability = 0.1 + Math.random() * 0.2 // 10-30%
        } else {
          probability = 0.3 + Math.random() * 0.4 // 30-70%
        }
      }

      return {
        id: i + 1,
        title: jobTitles[Math.floor(Math.random() * jobTitles.length)],
        company: companies[Math.floor(Math.random() * companies.length)],
        location: locations[Math.floor(Math.random() * locations.length)],
        prediction: i < fraudulentJobs ? "Fraudulent" : "Genuine",
        probability: probability,
        confidence: 0.8 + Math.random() * 0.2,
      }
    })

    // Calculate risk breakdown
    const lowRisk = mockResults.filter((r) => r.probability < 0.3).length
    const mediumRisk = mockResults.filter((r) => r.probability >= 0.3 && r.probability < 0.7).length
    const highRisk = mockResults.filter((r) => r.probability >= 0.7).length

    // Calculate probability statistics
    const probabilities = mockResults.map((r) => r.probability)
    const meanProb = probabilities.reduce((a, b) => a + b, 0) / probabilities.length
    const sortedProbs = [...probabilities].sort((a, b) => a - b)
    const medianProb = sortedProbs[Math.floor(sortedProbs.length / 2)]

    const comprehensiveResults = {
      type: "csv",
      filename: file.name,
      results: mockResults,
      summary: {
        total: totalJobs,
        genuine: genuineJobs,
        fraudulent: fraudulentJobs,
        fraud_rate: (fraudulentJobs / totalJobs) * 100,
        risk_breakdown: {
          low_risk: lowRisk,
          medium_risk: mediumRisk,
          high_risk: highRisk,
          low_risk_percentage: (lowRisk / totalJobs) * 100,
          medium_risk_percentage: (mediumRisk / totalJobs) * 100,
          high_risk_percentage: (highRisk / totalJobs) * 100,
        },
        probability_stats: {
          mean: meanProb,
          median: medianProb,
          min: Math.min(...probabilities),
          max: Math.max(...probabilities),
        },
      },
    }

    localStorage.setItem("predictionData", JSON.stringify(comprehensiveResults))
    router.push("/results")
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
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">CSV Upload & Analysis</h1>
          </div>
          <ThemeToggle />
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload CSV File</CardTitle>
              <CardDescription>Upload a CSV file containing job postings for batch analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                    : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500"
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                {file ? (
                  <div>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">{file.name}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      {isDragActive ? "Drop the CSV file here" : "Drag & drop a CSV file here"}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">or click to select a file</p>
                  </div>
                )}
              </div>

              {file && (
                <div className="mt-6 flex justify-center">
                  <Button onClick={handleAnalyze} disabled={loading} size="lg">
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing CSV...
                      </>
                    ) : (
                      <>
                        <FileText className="mr-2 h-4 w-4" />
                        Analyze CSV File
                      </>
                    )}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* CSV Format Guide */}
          <Card>
            <CardHeader>
              <CardTitle>CSV Format Requirements</CardTitle>
              <CardDescription>Your CSV file should contain the following columns</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3 text-gray-900 dark:text-white">Required Columns:</h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">title</code> - Job title
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">location</code> - Job location
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">description</code> - Job description
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">company_profile</code> - Company
                      info
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">requirements</code> - Job
                      requirements
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">benefits</code> - Benefits offered
                    </li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3 text-gray-900 dark:text-white">Optional Columns:</h4>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">employment_type</code> - Employment
                      type
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">required_experience</code> -
                      Experience level
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">required_education</code> -
                      Education level
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">industry</code> - Industry sector
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">telecommuting</code> - Remote work
                      (1/0)
                    </li>
                    <li>
                      • <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">has_company_logo</code> - Has logo
                      (1/0)
                    </li>
                  </ul>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Download className="h-4 w-4 text-blue-600" />
                  <span className="font-medium text-blue-900 dark:text-blue-100">Sample CSV Template</span>
                </div>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Download our sample CSV template to ensure your file is formatted correctly.
                </p>
                <Button variant="outline" size="sm" className="mt-2">
                  <Download className="mr-2 h-4 w-4" />
                  Download Template
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
