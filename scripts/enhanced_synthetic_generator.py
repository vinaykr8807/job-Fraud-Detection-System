#!/usr/bin/env python3
"""
Enhanced Synthetic Dataset Generator for Job Fraud Detection
Generates a large, realistic synthetic dataset for training ML models
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import json

class EnhancedSyntheticJobGenerator:
    def __init__(self):
        # Enhanced job titles by category with more variety
        self.job_titles = {
            'tech': [
                'Software Engineer', 'Senior Software Engineer', 'Lead Software Engineer',
                'Data Scientist', 'Senior Data Scientist', 'Machine Learning Engineer',
                'Web Developer', 'Frontend Developer', 'Backend Developer', 'Full Stack Developer',
                'Mobile App Developer', 'iOS Developer', 'Android Developer',
                'DevOps Engineer', 'Cloud Engineer', 'Site Reliability Engineer',
                'Cybersecurity Analyst', 'Information Security Specialist', 'Security Engineer',
                'Database Administrator', 'Data Engineer', 'Data Analyst',
                'Systems Administrator', 'Network Engineer', 'Infrastructure Engineer',
                'QA Engineer', 'Test Automation Engineer', 'Quality Assurance Specialist',
                'Product Manager', 'Technical Product Manager', 'Engineering Manager',
                'Solutions Architect', 'Cloud Architect', 'Software Architect',
                'UI/UX Designer', 'User Experience Designer', 'Product Designer'
            ],
            'business': [
                'Marketing Manager', 'Digital Marketing Manager', 'Content Marketing Manager',
                'Sales Representative', 'Account Executive', 'Business Development Representative',
                'Business Analyst', 'Senior Business Analyst', 'Data Analyst',
                'Project Manager', 'Program Manager', 'Scrum Master',
                'Account Manager', 'Customer Success Manager', 'Client Relations Manager',
                'Operations Manager', 'Operations Analyst', 'Process Improvement Specialist',
                'Product Manager', 'Product Marketing Manager', 'Brand Manager',
                'Strategy Consultant', 'Management Consultant', 'Business Consultant',
                'Marketing Coordinator', 'Marketing Specialist', 'Social Media Manager',
                'Sales Manager', 'Regional Sales Manager', 'Inside Sales Representative'
            ],
            'finance': [
                'Financial Analyst', 'Senior Financial Analyst', 'Investment Analyst',
                'Accountant', 'Senior Accountant', 'Staff Accountant',
                'Investment Banker', 'Investment Associate', 'Portfolio Manager',
                'Financial Advisor', 'Wealth Management Advisor', 'Financial Planner',
                'Risk Analyst', 'Credit Risk Analyst', 'Market Risk Analyst',
                'Credit Analyst', 'Loan Officer', 'Underwriter',
                'Tax Specialist', 'Tax Accountant', 'Tax Manager',
                'Auditor', 'Internal Auditor', 'External Auditor',
                'Controller', 'Assistant Controller', 'Finance Manager',
                'Treasury Analyst', 'Cash Management Specialist',
                'Compliance Officer', 'Regulatory Affairs Specialist',
                'Insurance Agent', 'Insurance Underwriter', 'Claims Adjuster'
            ],
            'healthcare': [
                'Registered Nurse', 'Licensed Practical Nurse', 'Nurse Practitioner',
                'Medical Assistant', 'Clinical Medical Assistant', 'Administrative Medical Assistant',
                'Physical Therapist', 'Occupational Therapist', 'Speech Therapist',
                'Pharmacist', 'Clinical Pharmacist', 'Hospital Pharmacist',
                'Medical Technician', 'Laboratory Technician', 'Radiology Technician',
                'Healthcare Administrator', 'Hospital Administrator', 'Clinic Manager',
                'Clinical Research Coordinator', 'Clinical Research Associate',
                'Medical Coder', 'Medical Billing Specialist', 'Health Information Technician',
                'Radiologic Technologist', 'MRI Technologist', 'CT Technologist',
                'Respiratory Therapist', 'Cardiovascular Technologist',
                'Physician Assistant', 'Medical Doctor', 'Specialist Physician'
            ],
            'education': [
                'Teacher', 'Elementary School Teacher', 'High School Teacher',
                'Professor', 'Assistant Professor', 'Associate Professor',
                'Academic Advisor', 'Student Success Coordinator', 'Admissions Counselor',
                'Curriculum Developer', 'Instructional Designer', 'Educational Consultant',
                'Education Coordinator', 'Program Coordinator', 'Academic Coordinator',
                'School Administrator', 'Principal', 'Vice Principal',
                'Librarian', 'Academic Librarian', 'Research Librarian',
                'Tutor', 'Private Tutor', 'Online Tutor',
                'Training Specialist', 'Corporate Trainer', 'Learning and Development Specialist'
            ],
            'other': [
                'Customer Service Representative', 'Customer Support Specialist', 'Call Center Agent',
                'HR Specialist', 'Human Resources Generalist', 'Recruiter',
                'Administrative Assistant', 'Executive Assistant', 'Office Manager',
                'Graphic Designer', 'Visual Designer', 'Creative Director',
                'Content Writer', 'Technical Writer', 'Copywriter',
                'Social Media Manager', 'Digital Marketing Specialist', 'SEO Specialist',
                'Receptionist', 'Front Desk Coordinator', 'Office Coordinator',
                'Research Assistant', 'Market Research Analyst', 'Survey Researcher',
                'Event Coordinator', 'Event Planner', 'Conference Manager',
                'Supply Chain Analyst', 'Logistics Coordinator', 'Warehouse Manager'
            ]
        }
        
        # Enhanced company profiles
        self.legitimate_companies = {
            'tech': [
                'TechCorp Solutions', 'Digital Innovations Inc', 'CloudFirst Technologies',
                'DataDriven Analytics', 'NextGen Software', 'Advanced Systems Group',
                'Enterprise Solutions Ltd', 'Innovation Labs', 'Future Technologies',
                'Quantum Computing Corp', 'AI Solutions Inc', 'Blockchain Innovations',
                'Cybersecurity Experts', 'Cloud Native Systems', 'DevOps Professionals'
            ],
            'consulting': [
                'Strategic Consulting Partners', 'Management Solutions Group', 'Business Excellence Corp',
                'Professional Services Inc', 'Advisory Solutions Ltd', 'Transformation Consultants',
                'Performance Improvement Group', 'Strategy & Operations', 'Change Management Experts'
            ],
            'finance': [
                'Financial Services Group', 'Investment Partners LLC', 'Wealth Management Corp',
                'Capital Advisors Inc', 'Risk Management Solutions', 'Asset Management Group',
                'Banking Solutions Ltd', 'Credit Union Services', 'Insurance Professionals'
            ],
            'healthcare': [
                'Healthcare Solutions Inc', 'Medical Services Group', 'Patient Care Corp',
                'Clinical Excellence Ltd', 'Health Systems Management', 'Medical Technology Inc',
                'Pharmaceutical Services', 'Diagnostic Solutions Group', 'Wellness Partners'
            ]
        }
        
        self.suspicious_companies = [
            'Quick Money Solutions', 'Work From Home Experts', 'Easy Cash Company',
            'Instant Success Ltd', 'Fast Track Careers', 'Dream Job Providers',
            'Ultimate Opportunity Corp', 'Guaranteed Income Inc', 'Perfect Job Match',
            'Amazing Careers Ltd', 'Incredible Opportunities', 'Money Making Experts',
            'Home Business Solutions', 'Financial Freedom Corp', 'Unlimited Earnings Inc',
            'Success Guaranteed LLC', 'Easy Money Systems', 'Work When You Want Inc',
            'Be Your Own Boss Ltd', 'Instant Wealth Solutions', 'No Experience Required Corp'
        ]
        
        # Enhanced locations with realistic distribution
        self.major_cities = [
            'New York, NY', 'San Francisco, CA', 'Los Angeles, CA', 'Chicago, IL',
            'Seattle, WA', 'Boston, MA', 'Austin, TX', 'Denver, CO',
            'Atlanta, GA', 'Dallas, TX', 'Philadelphia, PA', 'Phoenix, AZ'
        ]
        
        self.secondary_cities = [
            'Portland, OR', 'Nashville, TN', 'Charlotte, NC', 'Tampa, FL',
            'Detroit, MI', 'Minneapolis, MN', 'San Diego, CA', 'Miami, FL',
            'Pittsburgh, PA', 'Cleveland, OH', 'Kansas City, MO', 'Salt Lake City, UT'
        ]
        
        # Enhanced employment details
        self.employment_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship']
        self.experience_levels = ['Entry level', 'Mid level', 'Senior level', 'Executive', 'Not Applicable']
        self.education_levels = ['High School', "Bachelor's Degree", "Master's Degree", 'PhD', 'Professional', 'Not Specified']
        
        # Enhanced industries
        self.industries = [
            'Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing',
            'Retail', 'Consulting', 'Media', 'Real Estate', 'Transportation',
            'Energy', 'Government', 'Non-profit', 'Hospitality', 'Construction',
            'Telecommunications', 'Automotive', 'Aerospace', 'Biotechnology', 'Entertainment'
        ]
        
        # Job functions
        self.functions = [
            'Engineering', 'Marketing', 'Sales', 'Finance', 'Operations',
            'Human Resources', 'Customer Service', 'Research', 'Administration',
            'Consulting', 'Design', 'Writing', 'Management', 'Support', 'Analytics'
        ]
        
        # Fraud indicators for realistic fraudulent jobs
        self.fraud_keywords = [
            'urgent', 'guaranteed', 'easy money', 'work from home', 'no experience',
            'make money fast', 'unlimited earning', 'be your own boss', 'instant',
            'amazing opportunity', 'exclusive', 'secret', 'breakthrough', 'revolutionary',
            'life-changing', 'incredible', 'unbelievable', 'too good to be true'
        ]
    
    def generate_legitimate_job(self, category=None):
        """Generate a realistic legitimate job posting"""
        if category is None:
            category = random.choice(list(self.job_titles.keys()))
        
        title = random.choice(self.job_titles[category])
        
        # Select company based on category
        if category in self.legitimate_companies:
            company = random.choice(self.legitimate_companies[category])
        else:
            company = random.choice([comp for comps in self.legitimate_companies.values() for comp in comps])
        
        # Location with realistic distribution (80% major cities, 20% secondary)
        location = random.choice(self.major_cities if random.random() < 0.8 else self.secondary_cities)
        
        # Generate realistic job description
        descriptions = [
            f"We are seeking a qualified {title} to join our growing team at {company}. The ideal candidate will have strong technical skills and relevant experience in {category}. This role offers excellent opportunities for professional growth and development in a collaborative environment.",
            
            f"Join our dynamic team as a {title} at {company}. We offer competitive compensation, comprehensive benefits, and the opportunity to work on challenging projects that make a real impact. The successful candidate will contribute to our mission of delivering exceptional results.",
            
            f"{company} is looking for an experienced {title} to contribute to our innovative projects and help drive our business forward. We value diversity, creativity, and continuous learning. This position offers the chance to work with cutting-edge technology and talented professionals.",
            
            f"Exciting opportunity for a {title} to work with our team at {company}. We are committed to fostering an inclusive workplace where every team member can thrive. The role involves collaborating with cross-functional teams to deliver high-quality solutions.",
            
            f"As a {title} at {company}, you will play a key role in our organization's success and have the opportunity to make a significant impact. We provide a supportive environment for professional development and career advancement."
        ]
        
        requirements = [
            f"Bachelor's degree in relevant field or equivalent experience. 2-5 years of experience in {category}. Strong communication and problem-solving skills. Proficiency in industry-standard tools and technologies.",
            
            f"Proven experience in {category} with demonstrated success in similar roles. Excellent analytical and technical skills. Ability to work independently and as part of a team. Strong attention to detail and commitment to quality.",
            
            f"Relevant degree and professional experience in {category}. Strong interpersonal skills and ability to collaborate effectively with diverse teams. Experience with project management methodologies and client interaction preferred.",
            
            f"Technical expertise in {category} with hands-on experience in relevant tools and platforms. Strong problem-solving abilities and analytical thinking. Excellent written and verbal communication skills. Continuous learning mindset and adaptability."
        ]
        
        benefits = [
            "Comprehensive health insurance including medical, dental, and vision coverage. 401(k) retirement plan with company matching. Paid time off and holidays. Professional development opportunities and tuition reimbursement. Flexible work arrangements.",
            
            "Competitive salary and performance-based bonuses. Full benefits package including health, dental, vision, and life insurance. Retirement savings plan with employer contribution. Paid vacation, sick leave, and personal days. Career advancement opportunities.",
            
            "Excellent compensation package with health and wellness benefits. Retirement planning assistance and financial counseling. Paid time off and flexible scheduling options. Training and development programs. Employee assistance program and wellness initiatives.",
            
            "Health benefits including medical, dental, vision, and mental health coverage. Retirement savings plan with company match. Paid holidays and vacation time. Professional development budget and conference attendance. Work-life balance initiatives.",
            
            "Competitive salary with annual reviews and merit increases. Comprehensive benefits including health insurance and retirement plan. Paid time off and sabbatical opportunities. Learning and development programs. Collaborative and inclusive work environment."
        ]
        
        return {
            'title': title,
            'location': location,
            'department': random.choice(self.functions),
            'company_profile': f"{company} is a leading organization in the {random.choice(self.industries).lower()} industry, committed to innovation, excellence, and creating value for our clients and stakeholders.",
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
        """Generate a realistic fraudulent job posting with red flags"""
        title = random.choice([item for sublist in self.job_titles.values() for item in sublist])
        company = random.choice(self.suspicious_companies)
        location = random.choice(['Remote', 'Work from Home', 'Anywhere', 'USA'] + self.major_cities)
        
        # Fraudulent job descriptions with multiple red flags
        descriptions = [
            f"URGENT HIRING! Make $5000+ per week working from home as a {title}! No experience required! Start immediately and be your own boss! This amazing opportunity won't last long - apply now!",
            
            f"Incredible opportunity to earn BIG MONEY fast! {title} position available with unlimited earning potential. Work your own hours, set your own schedule. No qualifications necessary - we provide everything!",
            
            f"GUARANTEED high income! {title} needed for exclusive work-from-home opportunity. Earn $100-500 per day with our proven system. No experience needed - just follow our simple steps!",
            
            f"Revolutionary {title} position with breakthrough earning potential! Join thousands of successful people making incredible money from home. Secret system revealed - limited time offer!",
            
            f"Life-changing opportunity! Work from home {title} job with unbelievable income potential. Earn $3000-8000 per month part-time! No experience required - instant approval guaranteed!"
        ]
        
        requirements = [
            "No experience required! Just need a computer and internet connection. Must be motivated to earn serious money! Age 18+ only. Serious inquiries only - this is a real opportunity!",
            
            "Anyone can do this job! No special skills needed - we train you for FREE! Just need enthusiasm and desire to succeed. Must be willing to follow our proven system exactly!",
            
            "Basic computer skills helpful but not required. Must be willing to work hard and earn big money! No degree necessary - we provide all training materials and support!",
            
            "Perfect for stay-at-home parents, students, retirees, or anyone wanting extra income! No experience needed - our system works for everyone! Must be coachable and ready to start!",
            
            "No qualifications necessary! We're looking for motivated individuals ready to change their lives. Must have access to computer and phone. Serious people only - no time wasters!"
        ]
        
        benefits = [
            "Unlimited earning potential! Work from anywhere! Be your own boss! No commute required! Flexible schedule - work when you want! Financial freedom awaits!",
            
            "Amazing income opportunity! Work part-time or full-time! No office politics! Set your own hours! Incredible support system! Start earning immediately!",
            
            "Outstanding compensation! Work-life balance guaranteed! No experience required! Start making money today! Join our success team! Life-changing opportunity!",
            
            "Fantastic benefits package! High commission rates! Work from home comfort! No selling required! Proven system that works! Unlimited growth potential!",
            
            "Incredible opportunity for financial independence! Flexible schedule! Work from anywhere in the world! No boss to answer to! Start your new life today!"
        ]
        
        return {
            'title': title,
            'location': location,
            'department': random.choice(self.functions),
            'company_profile': f"{company} offers incredible opportunities for financial success, work-life balance, and personal freedom. Join thousands of successful people who have changed their lives with our proven system.",
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
    
    def generate_dataset(self, total_samples=75000, fraud_percentage=0.048):
        """Generate a comprehensive synthetic dataset"""
        print("🚀 ENHANCED SYNTHETIC DATASET GENERATOR")
        print("=" * 60)
        print(f"📊 Generating {total_samples:,} job postings...")
        print(f"📊 Target fraud rate: {fraud_percentage*100:.1f}%")
        
        fraudulent_count = int(total_samples * fraud_percentage)
        legitimate_count = total_samples - fraudulent_count
        
        print(f"📊 Legitimate jobs: {legitimate_count:,}")
        print(f"📊 Fraudulent jobs: {fraudulent_count:,}")
        
        jobs = []
        
        # Generate legitimate jobs with category distribution
        categories = list(self.job_titles.keys())
        category_weights = [0.35, 0.25, 0.15, 0.15, 0.10]  # tech, business, finance, healthcare, education, other
        
        print("\n🔨 Generating legitimate job postings...")
        for i in range(legitimate_count):
            category = np.random.choice(categories, p=category_weights)
            jobs.append(self.generate_legitimate_job(category))
            
            if (i + 1) % 5000 == 0:
                print(f"   ✅ Generated {i + 1:,} legitimate jobs...")
        
        # Generate fraudulent jobs
        print("\n🚨 Generating fraudulent job postings...")
        for i in range(fraudulent_count):
            jobs.append(self.generate_fraudulent_job())
            
            if (i + 1) % 500 == 0:
                print(f"   ✅ Generated {i + 1:,} fraudulent jobs...")
        
        # Shuffle the dataset
        print("\n🔀 Shuffling dataset...")
        random.shuffle(jobs)
        
        # Create DataFrame
        print("📋 Creating DataFrame...")
        df = pd.DataFrame(jobs)
        
        # Add additional realistic features
        print("🔧 Adding additional features...")
        df['salary_range'] = df.apply(self._generate_salary_range, axis=1)
        df['posted_date'] = df.apply(lambda x: self._generate_posted_date(), axis=1)
        df['application_deadline'] = df.apply(lambda x: self._generate_deadline(x['posted_date']), axis=1)
        df['job_id'] = df.apply(lambda x: self._generate_job_id(), axis=1)
        
        return df
    
    def _generate_salary_range(self, row):
        """Generate realistic salary ranges"""
        if row['fraudulent'] == 1:
            # Fraudulent jobs often have unrealistic salaries or vague promises
            if random.random() < 0.4:
                return f"${random.randint(80, 200)}K-${random.randint(300, 500)}K"
            elif random.random() < 0.3:
                return "Unlimited earning potential"
            else:
                return ""
        else:
            # Legitimate salary ranges based on experience level
            if random.random() < 0.75:  # 75% have salary info
                salary_ranges = {
                    'Entry level': (35000, 65000),
                    'Mid level': (55000, 95000),
                    'Senior level': (85000, 140000),
                    'Executive': (130000, 300000),
                    'Not Applicable': (25000, 50000)
                }
                
                min_sal, max_sal = salary_ranges.get(row['required_experience'], (40000, 80000))
                
                # Add some variation
                min_salary = min_sal + random.randint(-5000, 5000)
                max_salary = max_sal + random.randint(-10000, 15000)
                
                return f"${min_salary:,}-${max_salary:,}"
            else:
                return ""
    
    def _generate_posted_date(self):
        """Generate realistic posting dates"""
        start_date = datetime.now() - timedelta(days=120)
        end_date = datetime.now() - timedelta(days=1)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def _generate_deadline(self, posted_date):
        """Generate application deadlines"""
        if random.random() < 0.7:  # 70% have deadlines
            posted = datetime.strptime(posted_date, '%Y-%m-%d')
            deadline = posted + timedelta(days=random.randint(7, 45))
            return deadline.strftime('%Y-%m-%d')
        else:
            return ""
    
    def _generate_job_id(self):
        """Generate unique job IDs"""
        return f"JOB-{random.randint(100000, 999999)}"
    
    def save_dataset(self, df, filename='Training_Dataset.csv'):
        """Save the dataset with comprehensive statistics"""
        print(f"\n💾 SAVING DATASET")
        print("=" * 60)
        
        df.to_csv(filename, index=False)
        
        # Calculate and display statistics
        total_samples = len(df)
        legitimate_jobs = len(df[df['fraudulent'] == 0])
        fraudulent_jobs = len(df[df['fraudulent'] == 1])
        fraud_rate = fraudulent_jobs / total_samples * 100
        
        # Category distribution
        category_dist = df['function'].value_counts()
        
        # Industry distribution
        industry_dist = df['industry'].value_counts()
        
        print(f"✅ Dataset saved as '{filename}'")
        print(f"📊 DATASET STATISTICS:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Legitimate jobs: {legitimate_jobs:,} ({legitimate_jobs/total_samples*100:.2f}%)")
        print(f"   Fraudulent jobs: {fraudulent_jobs:,} ({fraud_rate:.2f}%)")
        print(f"   File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        
        print(f"\n📊 TOP JOB FUNCTIONS:")
        for func, count in category_dist.head().items():
            print(f"   {func}: {count:,} ({count/total_samples*100:.1f}%)")
        
        print(f"\n📊 TOP INDUSTRIES:")
        for industry, count in industry_dist.head().items():
            print(f"   {industry}: {count:,} ({count/total_samples*100:.1f}%)")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_samples': total_samples,
            'legitimate_jobs': legitimate_jobs,
            'fraudulent_jobs': fraudulent_jobs,
            'fraud_rate': fraud_rate,
            'features': list(df.columns),
            'category_distribution': category_dist.to_dict(),
            'industry_distribution': industry_dist.to_dict()
        }
        
        with open(filename.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Metadata saved as '{filename.replace('.csv', '_metadata.json')}'")

def main():
    """Generate enhanced synthetic training dataset"""
    print("🎯 ENHANCED SYNTHETIC JOB DATASET GENERATOR")
    print("=" * 80)
    print("This generator creates a large, realistic dataset for training")
    print("fraud detection models with improved diversity and realism.")
    print("=" * 80)
    
    generator = EnhancedSyntheticJobGenerator()
    
    # Generate comprehensive dataset
    dataset = generator.generate_dataset(
        total_samples=75000,  # 75K samples for robust training
        fraud_percentage=0.048  # 4.8% fraud rate (realistic)
    )
    
    # Save the dataset
    generator.save_dataset(dataset, 'Training_Dataset.csv')
    
    print("\n🎉 DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print("📝 NEXT STEPS:")
    print("   1. Run: python scripts/comprehensive_model_trainer.py")
    print("   2. This will train multiple ML models with SMOTE")
    print("   3. Models will be evaluated and compared")
    print("   4. Best model will be saved for predictions")
    print("   5. Use universal_csv_analyzer.py for analyzing uploaded datasets")
    print("=" * 80)

if __name__ == "__main__":
    main()
