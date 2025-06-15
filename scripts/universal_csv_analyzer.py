#!/usr/bin/env python3
"""
Universal CSV Analyzer for Job Fraud Detection
Processes any uploaded CSV dataset and generates predictions using trained models
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

class UniversalCSVAnalyzer:
    def __init__(self, model_path='model_outputs/fraud_detection_model_system.pkl'):
        self.model_path = model_path
        self.model_system = None
        self.best_model = None
        self.vectorizer = None
        self.feature_names = None
        
        # Download NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        nltk_downloads = ['stopwords', 'punkt', 'wordnet', 'brown', 'punkt_tab']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                pass
    
    def load_trained_models(self):
        """Load the pre-trained model system"""
        print("🔄 Loading trained model system...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model system not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model_system = pickle.load(f)
        
        self.best_model = self.model_system['best_model']
        self.vectorizer = self.model_system['vectorizer']
        self.feature_names = self.model_system['feature_names']
        
        print(f"✅ Model system loaded successfully!")
        print(f"🏆 Best model: {self.model_system['best_model_name']}")
        print(f"📊 Available models: {list(self.model_system['all_models'].keys())}")
        
        return True
    
    def analyze_csv_structure(self, df):
        """Analyze CSV structure and map columns"""
        print(f"\n📊 ANALYZING CSV STRUCTURE")
        print("=" * 50)
        print(f"📋 Dataset shape: {df.shape}")
        print(f"📋 Columns found: {list(df.columns)}")
        
        # Column mapping for different naming conventions
        column_mapping = {
            'title': ['title', 'job_title', 'position', 'role', 'job_name'],
            'description': ['description', 'job_description', 'details', 'summary', 'overview'],
            'company_profile': ['company_profile', 'company', 'company_name', 'employer', 'organization'],
            'location': ['location', 'city', 'address', 'place', 'region'],
            'requirements': ['requirements', 'qualifications', 'skills', 'experience_required'],
            'benefits': ['benefits', 'perks', 'compensation', 'package'],
            'employment_type': ['employment_type', 'job_type', 'type', 'category'],
            'telecommuting': ['telecommuting', 'remote', 'work_from_home', 'wfh'],
            'has_company_logo': ['has_company_logo', 'logo', 'company_logo']
        }
        
        # Find matching columns
        mapped_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if any(possible.lower() in col.lower() for possible in possible_names):
                    mapped_columns[standard_name] = col
                    break
        
        print(f"📋 Mapped columns: {mapped_columns}")
        
        # Check for required columns
        required_columns = ['title', 'description']
        missing_required = [col for col in required_columns if col not in mapped_columns]
        
        if missing_required:
            print(f"⚠️  Missing required columns: {missing_required}")
            print("📝 Will attempt to use available text columns...")
        
        return mapped_columns
    
    def preprocess_csv_data(self, df, column_mapping):
        """Preprocess CSV data for prediction"""
        print(f"\n🔄 PREPROCESSING CSV DATA")
        print("=" * 50)
        
        processed_data = []
        
        for idx, row in df.iterrows():
            # Extract text fields
            title = str(row.get(column_mapping.get('title', ''), '')).strip()
            description = str(row.get(column_mapping.get('description', ''), '')).strip()
            company = str(row.get(column_mapping.get('company_profile', ''), '')).strip()
            location = str(row.get(column_mapping.get('location', ''), '')).strip()
            requirements = str(row.get(column_mapping.get('requirements', ''), '')).strip()
            benefits = str(row.get(column_mapping.get('benefits', ''), '')).strip()
            
            # If no description, try to use other text fields
            if not description:
                description = ' '.join([title, company, requirements, benefits]).strip()
            
            # If still no meaningful text, skip this row
            if len(description) < 10:
                print(f"⚠️  Skipping row {idx+1}: Insufficient text data")
                continue
            
            # Create combined text
            combined_text = ' '.join([title, location, company, description, requirements, benefits])
            
            # Preprocess text
            processed_text = self._preprocess_text(combined_text)
            
            # Extract numerical features
            telecommuting = 0
            has_logo = 0
            
            if 'telecommuting' in column_mapping:
                tel_val = str(row.get(column_mapping['telecommuting'], '0')).lower()
                telecommuting = 1 if tel_val in ['1', 'true', 'yes', 'remote'] else 0
            
            if 'has_company_logo' in column_mapping:
                logo_val = str(row.get(column_mapping['has_company_logo'], '0')).lower()
                has_logo = 1 if logo_val in ['1', 'true', 'yes'] else 0
            
            processed_data.append({
                'original_index': idx,
                'title': title[:100],
                'company': company[:50] if company else 'Unknown',
                'location': location[:50] if location else 'Not specified',
                'processed_text': processed_text,
                'text_length': len(processed_text),
                'word_count': len(processed_text.split()),
                'telecommuting': telecommuting,
                'has_company_logo': has_logo
            })
            
            if (len(processed_data)) % 100 == 0:
                print(f"✅ Processed {len(processed_data)} rows...")
        
        print(f"✅ Successfully processed {len(processed_data)} out of {len(df)} rows")
        return processed_data
    
    def _preprocess_text(self, text):
        """Preprocess text using the same pipeline as training"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            text = ' '.join([word for word in words if word not in stop_words and len(word) > 2])
        except:
            pass
        
        # Lemmatization
        try:
            lemmatizer = WordNetLemmatizer()
            words = word_tokenize(text)
            text = ' '.join([lemmatizer.lemmatize(word) for word in words])
        except:
            pass
        
        return text
    
    def generate_predictions(self, processed_data):
        """Generate fraud predictions for processed data"""
        print(f"\n🔮 GENERATING PREDICTIONS")
        print("=" * 50)
        
        if not processed_data:
            raise ValueError("No valid data to process")
        
        # Extract text for vectorization
        texts = [item['processed_text'] for item in processed_data]
        
        # Vectorize text
        print("🔢 Vectorizing text features...")
        text_features = self.vectorizer.transform(texts).toarray()
        
        # Prepare feature matrix
        feature_matrix = []
        for i, item in enumerate(processed_data):
            # Combine text features with numerical features
            row_features = list(text_features[i]) + [
                item['text_length'],
                item['word_count'],
                item['telecommuting'],
                item['has_company_logo']
            ]
            feature_matrix.append(row_features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Ensure feature matrix has correct number of features
        expected_features = len(self.feature_names)
        actual_features = feature_matrix.shape[1]
        
        if actual_features != expected_features:
            print(f"⚠️  Feature mismatch: expected {expected_features}, got {actual_features}")
            # Pad or truncate as needed
            if actual_features < expected_features:
                padding = np.zeros((feature_matrix.shape[0], expected_features - actual_features))
                feature_matrix = np.hstack([feature_matrix, padding])
            else:
                feature_matrix = feature_matrix[:, :expected_features]
        
        # Generate predictions
        print("🎯 Generating predictions...")
        predictions = self.best_model.predict(feature_matrix)
        probabilities = self.best_model.predict_proba(feature_matrix)[:, 1]
        
        # Combine results
        results = []
        for i, item in enumerate(processed_data):
            results.append({
                'id': i + 1,
                'original_index': item['original_index'],
                'title': item['title'],
                'company': item['company'],
                'location': item['location'],
                'prediction': 'Fraudulent' if predictions[i] == 1 else 'Genuine',
                'probability': float(probabilities[i]),
                'confidence': float(max(self.best_model.predict_proba(feature_matrix[i:i+1])[0]))
            })
        
        print(f"✅ Generated predictions for {len(results)} job postings")
        return results
    
    def calculate_comprehensive_statistics(self, results):
        """Calculate comprehensive statistics from results"""
        print(f"\n📊 CALCULATING STATISTICS")
        print("=" * 50)
        
        total = len(results)
        fraudulent = len([r for r in results if r['prediction'] == 'Fraudulent'])
        genuine = total - fraudulent
        
        # Risk categorization
        low_risk = len([r for r in results if r['probability'] < 0.3])
        medium_risk = len([r for r in results if 0.3 <= r['probability'] < 0.7])
        high_risk = len([r for r in results if r['probability'] >= 0.7])
        
        # Probability statistics
        probabilities = [r['probability'] for r in results]
        mean_prob = np.mean(probabilities)
        median_prob = np.median(probabilities)
        
        # Create detailed probability distribution
        prob_distribution = []
        for i in range(20):  # 20 bins
            start = i * 0.05
            end = (i + 1) * 0.05
            count = len([p for p in probabilities if start <= p < end])
            prob_distribution.append({
                'range': f"{start*100:.0f}-{end*100:.0f}%",
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            })
        
        summary = {
            'total': total,
            'genuine': genuine,
            'fraudulent': fraudulent,
            'fraud_rate': (fraudulent / total * 100) if total > 0 else 0,
            'risk_breakdown': {
                'low_risk': low_risk,
                'low_risk_percentage': (low_risk / total * 100) if total > 0 else 0,
                'medium_risk': medium_risk,
                'medium_risk_percentage': (medium_risk / total * 100) if total > 0 else 0,
                'high_risk': high_risk,
                'high_risk_percentage': (high_risk / total * 100) if total > 0 else 0
            },
            'probability_stats': {
                'mean': float(mean_prob),
                'median': float(median_prob),
                'min': float(min(probabilities)) if probabilities else 0,
                'max': float(max(probabilities)) if probabilities else 0,
                'std': float(np.std(probabilities)) if probabilities else 0
            },
            'probability_distribution': prob_distribution
        }
        
        print(f"📊 Analysis Summary:")
        print(f"   Total jobs: {total:,}")
        print(f"   Genuine: {genuine:,} ({genuine/total*100:.1f}%)")
        print(f"   Fraudulent: {fraudulent:,} ({fraudulent/total*100:.1f}%)")
        print(f"   High risk: {high_risk:,} ({high_risk/total*100:.1f}%)")
        print(f"   Mean probability: {mean_prob*100:.1f}%")
        
        return summary
    
    def analyze_csv_file(self, csv_path, output_path=None):
        """Main function to analyze a CSV file"""
        print("🚀 UNIVERSAL CSV ANALYZER")
        print("=" * 60)
        print(f"📁 Input file: {csv_path}")
        
        if output_path is None:
            output_path = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Load CSV
            print(f"\n📂 Loading CSV file...")
            df = pd.read_csv(csv_path)
            
            # Analyze structure
            column_mapping = self.analyze_csv_structure(df)
            
            # Preprocess data
            processed_data = self.preprocess_csv_data(df, column_mapping)
            
            if not processed_data:
                raise ValueError("No valid data could be processed from the CSV file")
            
            # Generate predictions
            results = self.generate_predictions(processed_data)
            
            # Calculate statistics
            summary = self.calculate_comprehensive_statistics(results)
            
            # Create final analysis result
            analysis_result = {
                'filename': os.path.basename(csv_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': self.model_system['best_model_name'],
                'original_rows': len(df),
                'processed_rows': len(processed_data),
                'column_mapping': column_mapping,
                'results': results,
                'summary': summary,
                'model_performance': self.model_system['model_performance']
            }
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📊 Results saved to: {output_path}")
            print(f"🎯 Model used: {self.model_system['best_model_name']}")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise

def main():
    """Main function for CSV analysis"""
    if len(sys.argv) < 2:
        print("Usage: python universal_csv_analyzer.py <csv_file_path> [output_file_path]")
        print("\nExample:")
        print("  python universal_csv_analyzer.py data/jobs.csv")
        print("  python universal_csv_analyzer.py data/jobs.csv results.json")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = UniversalCSVAnalyzer()
        
        # Load trained models
        analyzer.load_trained_models()
        
        # Analyze CSV file
        analyzer.analyze_csv_file(csv_path, output_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
