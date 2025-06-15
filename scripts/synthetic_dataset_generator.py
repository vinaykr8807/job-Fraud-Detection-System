#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Job Fraud Detection
This script generates a large synthetic dataset for training the fraud detection model
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

class SyntheticJobDataGenerator:
    def __init__(self):
        # Job titles by category
        self.job_titles = {
            'tech': [
                'Software Engineer', 'Data Scientist', 'Web Developer', 'Mobile App Developer',
                'DevOps Engineer', 'Machine Learning Engineer', 'Full Stack Developer',
                'Frontend Developer', 'Backend Developer', 'Cloud Architect', 'Cybersecurity Analyst',
                'Database Administrator', 'Systems Administrator', 'Network Engineer', 'QA Engineer'
            ],
            'business': [
                'Marketing Manager', 'Sales Representative', 'Business Analyst', 'Project Manager',
                'Account Manager', 'Operations Manager', 'Product Manager', 'Business Development Manager',
                'Marketing Coordinator', 'Sales Manager', 'Customer Success Manager', 'Strategy Consultant'
            ],
            'finance': [
                'Financial Analyst', 'Accountant', 'Investment Banker', 'Financial Advisor',
                'Risk Analyst', 'Credit Analyst', 'Tax Specialist', 'Auditor', 'Controller',
                'Treasury Analyst', 'Compliance Officer', 'Insurance Agent'
            ],
            'healthcare': [
                'Registered Nurse', 'Medical Assistant', 'Physical Therapist', 'Pharmacist',
                'Medical Technician', 'Healthcare Administrator', 'Clinical Research Coordinator',
                'Medical Coder', 'Radiologic Technologist', 'Respiratory Therapist'
            ],
            'education': [
                'Teacher', 'Professor', 'Academic Advisor', 'Curriculum Developer',
                'Education Coordinator', 'School Administrator', 'Librarian', 'Tutor',
                'Training Specialist', 'Instructional Designer'
            ],
            'other': [
                'Customer Service Representative', 'HR Specialist', 'Administrative Assistant',
                'Graphic Designer', 'Content Writer', 'Social Media Manager', 'Receptionist',
                'Office Manager', 'Executive Assistant', 'Research Assistant'
            ]
        }
        
        # Company names and profiles
        self.companies = {
            'legitimate': [
                'TechCorp Solutions', 'Global Innovations Inc', 'DataDriven Analytics',
                'CloudFirst Technologies', 'NextGen Software', 'Digital Transformation Co',
                'Enterprise Solutions Ltd', 'Innovation Labs', 'Future Systems Inc',
                'Advanced Technologies Group', 'Strategic Consulting Partners',
                'Professional Services Corp', 'Industry Leaders LLC'
            ],
            'suspicious': [
                'Quick Money Solutions', 'Work From Home Experts', 'Easy Cash Company',
                'Instant Success Ltd', 'Fast Track Careers', 'Dream Job Providers',
                'Ultimate Opportunity Corp', 'Guaranteed Income Inc', 'Perfect Job Match',
                'Amazing Careers Ltd', 'Incredible Opportunities'
            ]
        }
        
        # Locations
        self.locations = [
            'New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Austin, TX',
            'Seattle, WA', 'Boston, MA', 'Los Angeles, CA', 'Denver, CO',
            'Miami, FL', 'Atlanta, GA', 'Portland, OR', 'Phoenix, AZ',
            'Dallas, TX', 'Philadelphia, PA', 'San Diego, CA', 'Minneapolis, MN',
            'Nashville, TN', 'Charlotte, NC', 'Tampa, FL', 'Detroit, MI'
        ]
        
        # Employment types
        self.employment_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
        
        # Experience levels
        self.experience_levels = ['Entry level', 'Mid level', 'Senior level', 'Executive', 'Not Applicable']
        
        # Education levels
        self.education_levels = [
            'High School', "Bachelor's Degree", "Master's Degree", 'PhD', 
            'Professional', 'Not Specified'
        ]
        
        # Industries
        self.industries = [
            'Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing',
            'Retail', 'Consulting', 'Media', 'Real Estate', 'Transportation',
            'Energy', 'Government', 'Non-profit', 'Hospitality', 'Construction'
        ]
        
        # Job functions
        self.functions = [
            'Engineering', 'Marketing', 'Sales', 'Finance', 'Operations',
            'Human Resources', 'Customer Service', 'Research', 'Administration',
            'Consulting', 'Design', 'Writing', 'Management', 'Support'
        ]
    
    def generate_legitimate_job(self):
        """Generate a legitimate job posting"""
        category = random.choice(list(self.job_titles.keys()))
        title = random.choice(self.job_titles[category])
        company = random.choice(self.companies['legitimate'])
        location = random.choice(self.locations)
        
        # Generate realistic job description
        descriptions = [
            f"We are seeking a qualified {title} to join our growing team. The ideal candidate will have strong technical skills and experience in the field.",
            f"Join our dynamic team as a {title}. We offer competitive salary, excellent benefits, and opportunities for professional growth.",
            f"Our company is looking for an experienced {title} to contribute to our innovative projects and help drive our business forward.",
            f"We have an exciting opportunity for a {title} to work with cutting-edge technology and collaborate with talented professionals.",
            f"As a {title}, you will play a key role in our organization's success and have the opportunity to make a significant impact."
        ]
        
        requirements = [
            f"Bachelor's degree in relevant field or equivalent experience. 2-5 years of experience in {category}. Strong communication and problem-solving skills.",
            f"Proven experience in {category}. Excellent analytical and technical skills. Ability to work independently and as part of a team.",
            f"Relevant degree and professional experience. Strong attention to detail and ability to meet deadlines. Proficiency in industry-standard tools.",
            f"Experience with modern technologies and methodologies. Strong interpersonal skills and ability to collaborate effectively.",
            f"Technical expertise in {category}. Experience with project management and client interaction. Continuous learning mindset."
        ]
        
        benefits = [
            "Health insurance, dental and vision coverage, 401(k) with company match, paid time off, professional development opportunities.",
            "Competitive salary, comprehensive benefits package, flexible work arrangements, career advancement opportunities.",
            "Medical, dental, and vision insurance, retirement savings plan, paid holidays, training and development programs.",
            "Health benefits, life insurance, disability coverage, employee assistance program, tuition reimbursement.",
            "Full benefits package including health insurance, retirement plan, paid vacation, sick leave, and professional development."
        ]
        
        return {
            'title': title,
            'location': location,
            'department': random.choice(self.functions),
            'company_profile': f"{company} is a leading company in the {random.choice(self.industries).lower()} industry, committed to innovation and excellence.",
            'description': random.choice(descriptions),
            'requirements': random.choice(requirements),
            'benefits': random.choice(benefits),
            'employment_type': random.choice(self.employment_types),
            'required_experience': random.choice(self.experience_levels),
            'required_education': random.choice(self.education_levels),
            'industry': random.choice(self.industries),
            'function': random.choice(self.functions),
            'telecommuting': random.choice([0, 1]),
            'has_company_logo': random.choice([0, 1]),
            'fraudulent': 0
        }
    
    def generate_fraudulent_job(self):
        """Generate a fraudulent job posting"""
        title = random.choice([item for sublist in self.job_titles.values() for item in sublist])
        company = random.choice(self.companies['suspicious'])
        location = random.choice(self.locations + ['Remote', 'Work from Home', 'Anywhere'])
        
        # Fraudulent job descriptions with red flags
        descriptions = [
            f"URGENT! Make $5000+ per week working from home as a {title}! No experience required! Start immediately!",
            f"Amazing opportunity to earn big money fast! {title} position available. Work your own hours, be your own boss!",
            f"Guaranteed high income! {title} needed. No qualifications necessary. Easy work, great pay!",
            f"Exclusive opportunity! {title} position with unlimited earning potential. Join now and start earning today!",
            f"Work from home {title} job. Earn $100-500 per day! No experience needed. Apply now for instant approval!"
        ]
        
        requirements = [
            "No experience required! Just need a computer and internet connection. Must be motivated to earn money!",
            "Anyone can do this job! No special skills needed. Just follow our simple system and start earning!",
            "Basic computer skills helpful but not required. Must be willing to work hard and earn big money!",
            "No degree necessary! We provide all training. Just need enthusiasm and desire to succeed!",
            "Perfect for stay-at-home parents, students, or anyone wanting extra income! No experience needed!"
        ]
        
        benefits = [
            "Unlimited earning potential! Work from anywhere! Be your own boss! No commute required!",
            "Flexible schedule, work when you want! High pay rates! Bonuses available! Start earning immediately!",
            "Amazing income opportunity! Work part-time or full-time! No office politics! Financial freedom!",
            "Incredible benefits package! High commission rates! Work from home! Set your own schedule!",
            "Outstanding compensation! Work-life balance! No experience required! Start making money today!"
        ]
        
        return {
            'title': title,
            'location': location,
            'department': random.choice(self.functions),
            'company_profile': f"{company} offers incredible opportunities for financial success and work-life balance.",
            'description': random.choice(descriptions),
            'requirements': random.choice(requirements),
            'benefits': random.choice(benefits),
            'employment_type': random.choice(['Full-time', 'Part-time', 'Contract', 'Flexible']),
            'required_experience': random.choice(['Not Applicable', 'Entry level']),
            'required_education': random.choice(['Not Specified', 'High School']),
            'industry': random.choice(self.industries),
            'function': random.choice(self.functions),
            'telecommuting': 1,  # Most fraudulent jobs claim remote work
            'has_company_logo': random.choice([0, 1]),
            'fraudulent': 1
        }
    
    def generate_dataset(self, total_samples=50000, fraud_percentage=0.05):
        """Generate a complete synthetic dataset"""
        print(f"Generating synthetic dataset with {total_samples} samples...")
        print(f"Fraud percentage: {fraud_percentage*100:.1f}%")
        
        fraudulent_count = int(total_samples * fraud_percentage)
        legitimate_count = total_samples - fraudulent_count
        
        print(f"Legitimate jobs: {legitimate_count}")
        print(f"Fraudulent jobs: {fraudulent_count}")
        
        jobs = []
        
        # Generate legitimate jobs
        for i in range(legitimate_count):
            jobs.append(self.generate_legitimate_job())
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} legitimate jobs...")
        
        # Generate fraudulent jobs
        for i in range(fraudulent_count):
            jobs.append(self.generate_fraudulent_job())
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} fraudulent jobs...")
        
        # Shuffle the dataset
        random.shuffle(jobs)
        
        # Create DataFrame
        df = pd.DataFrame(jobs)
        
        # Add some additional features
        df['salary_range'] = df.apply(self._generate_salary_range, axis=1)
        df['posted_date'] = df.apply(lambda x: self._generate_posted_date(), axis=1)
        
        return df
    
    def _generate_salary_range(self, row):
        """Generate realistic salary ranges"""
        if row['fraudulent'] == 1:
            # Fraudulent jobs often have unrealistic salaries
            if random.random() < 0.3:
                return f"{random.randint(80, 200)}-{random.randint(300, 500)}"
            else:
                return ""
        else:
            # Legitimate salary ranges
            if random.random() < 0.7:  # 70% have salary info
                base_salary = {
                    'Entry level': random.randint(35, 55),
                    'Mid level': random.randint(55, 85),
                    'Senior level': random.randint(85, 130),
                    'Executive': random.randint(130, 250),
                    'Not Applicable': random.randint(25, 45)
                }.get(row['required_experience'], random.randint(40, 80))
                
                min_salary = base_salary
                max_salary = int(base_salary * 1.3)
                return f"{min_salary}-{max_salary}"
            else:
                return ""
    
    def _generate_posted_date(self):
        """Generate realistic posting dates"""
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def save_dataset(self, df, filename='Training_Dataset.csv'):
        """Save the dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"\n✅ Dataset saved as '{filename}'")
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Legitimate jobs: {len(df[df['fraudulent'] == 0])}")
        print(f"   Fraudulent jobs: {len(df[df['fraudulent'] == 1])}")
        print(f"   Fraud rate: {len(df[df['fraudulent'] == 1])/len(df)*100:.2f}%")
        print(f"   File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

def main():
    """Generate synthetic training dataset"""
    print("🚀 Synthetic Job Dataset Generator")
    print("=" * 50)
    
    generator = SyntheticJobDataGenerator()
    
    # Generate large dataset for training
    dataset = generator.generate_dataset(
        total_samples=50000,  # 50K samples for robust training
        fraud_percentage=0.05  # 5% fraud rate (realistic)
    )
    
    # Save the dataset
    generator.save_dataset(dataset, 'Training_Dataset.csv')
    
    print("\n🎯 Dataset Generation Complete!")
    print("📝 Next Steps:")
    print("   1. Run: python scripts/enhanced_model_trainer.py")
    print("   2. This will train your fraud detection model")
    print("   3. Users can then get predictions without seeing training data")

if __name__ == "__main__":
    main()
