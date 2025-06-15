"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ArrowLeft, Send, Loader2 } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"
import { useRouter } from "next/navigation"

interface JobFormData {
  title: string
  location: string
  company_profile: string
  description: string
  requirements: string
  benefits: string
  employment_type: string
  required_experience: string
  required_education: string
  industry: string
  function: string
  telecommuting: number
  has_company_logo: number
}

export default function PredictFormPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState<JobFormData>({
    title: "",
    location: "",
    company_profile: "",
    description: "",
    requirements: "",
    benefits: "",
    employment_type: "",
    required_experience: "",
    required_education: "",
    industry: "",
    function: "",
    telecommuting: 0,
    has_company_logo: 0,
  })

  const handleInputChange = (field: keyof JobFormData, value: string | number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Store form data in localStorage for the results page
    localStorage.setItem(
      "predictionData",
      JSON.stringify({
        type: "single",
        data: formData,
        result: {
          prediction: Math.random() > 0.7 ? "Fraudulent" : "Genuine",
          probability: Math.random(),
          confidence: 0.85 + Math.random() * 0.15,
        },
      }),
    )

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
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Job Analysis Form</h1>
          </div>
          <ThemeToggle />
        </div>

        <Card className="max-w-4xl mx-auto">
          <CardHeader>
            <CardTitle>Enter Job Details for Analysis</CardTitle>
            <CardDescription>
              Fill out the form below to analyze a job posting for potential fraud indicators
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="title">Job Title *</Label>
                  <Input
                    id="title"
                    value={formData.title}
                    onChange={(e) => handleInputChange("title", e.target.value)}
                    placeholder="e.g., Software Engineer"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="location">Location *</Label>
                  <Input
                    id="location"
                    value={formData.location}
                    onChange={(e) => handleInputChange("location", e.target.value)}
                    placeholder="e.g., New York, NY"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="company_profile">Company Profile</Label>
                <Textarea
                  id="company_profile"
                  value={formData.company_profile}
                  onChange={(e) => handleInputChange("company_profile", e.target.value)}
                  placeholder="Brief description of the hiring company..."
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Job Description *</Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => handleInputChange("description", e.target.value)}
                  placeholder="Detailed job description..."
                  rows={4}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="requirements">Requirements</Label>
                <Textarea
                  id="requirements"
                  value={formData.requirements}
                  onChange={(e) => handleInputChange("requirements", e.target.value)}
                  placeholder="Required qualifications and skills..."
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="benefits">Benefits</Label>
                <Textarea
                  id="benefits"
                  value={formData.benefits}
                  onChange={(e) => handleInputChange("benefits", e.target.value)}
                  placeholder="Offered benefits and perks..."
                  rows={3}
                />
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="employment_type">Employment Type</Label>
                  <Select onValueChange={(value) => handleInputChange("employment_type", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select employment type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Full-time">Full-time</SelectItem>
                      <SelectItem value="Part-time">Part-time</SelectItem>
                      <SelectItem value="Contract">Contract</SelectItem>
                      <SelectItem value="Temporary">Temporary</SelectItem>
                      <SelectItem value="Internship">Internship</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="required_experience">Required Experience</Label>
                  <Select onValueChange={(value) => handleInputChange("required_experience", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select experience level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Entry level">Entry level</SelectItem>
                      <SelectItem value="Mid level">Mid level</SelectItem>
                      <SelectItem value="Senior level">Senior level</SelectItem>
                      <SelectItem value="Executive">Executive</SelectItem>
                      <SelectItem value="Not Applicable">Not Applicable</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="required_education">Required Education</Label>
                  <Select onValueChange={(value) => handleInputChange("required_education", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select education level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="High School">High School</SelectItem>
                      <SelectItem value="Bachelor's Degree">Bachelor's Degree</SelectItem>
                      <SelectItem value="Master's Degree">Master's Degree</SelectItem>
                      <SelectItem value="PhD">PhD</SelectItem>
                      <SelectItem value="Professional">Professional</SelectItem>
                      <SelectItem value="Not Specified">Not Specified</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="industry">Industry</Label>
                  <Input
                    id="industry"
                    value={formData.industry}
                    onChange={(e) => handleInputChange("industry", e.target.value)}
                    placeholder="e.g., Technology, Healthcare"
                  />
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="function">Job Function</Label>
                  <Input
                    id="function"
                    value={formData.function}
                    onChange={(e) => handleInputChange("function", e.target.value)}
                    placeholder="e.g., Engineering, Marketing"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="telecommuting">Remote Work Allowed</Label>
                  <Select onValueChange={(value) => handleInputChange("telecommuting", Number.parseInt(value))}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select option" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Yes</SelectItem>
                      <SelectItem value="0">No</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="has_company_logo">Company Has Logo</Label>
                <Select onValueChange={(value) => handleInputChange("has_company_logo", Number.parseInt(value))}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Yes</SelectItem>
                    <SelectItem value="0">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing Job Posting...
                  </>
                ) : (
                  <>
                    <Send className="mr-2 h-4 w-4" />
                    Analyze Job Posting
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
