#!/usr/bin/env python3
"""
Enhanced Job Fraud Detection Model Trainer
This script trains multiple models and creates a comprehensive analysis system
"""

import os
import sys
import warnings
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import ADASYN
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.style.use('default')

class ComprehensiveFraudDetector:
    def __init__(self, dataset_path="Training Dataset.csv"):
        self.dataset_path = dataset_path
        self.models = {}
        self.vectorizer = None
        self.selector = None
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        
        # Download NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        nltk_downloads = ['stopwords', 'punkt', 'wordnet', 'brown', 'punkt_tab']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                print(f"Warning: Could not download {item}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the training dataset"""
        print("Loading training dataset...")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Training dataset not found: {self.dataset_path}")
        
        # Load data
        self.raw_data = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {self.raw_data.shape}")
        
        # Basic preprocessing
        self.clean_data = self.raw_data.copy()
        self.clean_data = self.clean_data.drop_duplicates()
        
        # Handle missing values
        text_columns = ['title', 'location', 'department', 'company_profile', 
                       'description', 'requirements', 'benefits', 'employment_type',
                       'required_experience', 'required_education', 'industry', 'function']
        
        for col in text_columns:
            if col in self.clean_data.columns:
                self.clean_data[col] = self.clean_data[col].fillna("")
        
        # Create combined text
        self.clean_data['combined_text'] = (
            self.clean_data.get('title', '').astype(str) + ' ' +
            self.clean_data.get('location', '').astype(str) + ' ' +
            self.clean_data.get('company_profile', '').astype(str) + ' ' +
            self.clean_data.get('description', '').astype(str) + ' ' +
            self.clean_data.get('requirements', '').astype(str) + ' ' +
            self.clean_data.get('benefits', '').astype(str)
        )
        
        # Preprocess text
        print("Preprocessing text data...")
        self.clean_data['processed_text'] = self.clean_data['combined_text'].apply(self._preprocess_text)
        
        # Handle numerical features
        if 'telecommuting' in self.clean_data.columns:
            self.clean_data['telecommuting'] = self.clean_data['telecommuting'].fillna(0)
        if 'has_company_logo' in self.clean_data.columns:
            self.clean_data['has_company_logo'] = self.clean_data['has_company_logo'].fillna(0)
        
        print(f"Preprocessed dataset shape: {self.clean_data.shape}")
        return self.clean_data
    
    def _preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        
        try:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            text = ' '.join([word for word in words if word not in stop_words and len(word) > 2])
        except:
            pass
        
        try:
            lemmatizer = WordNetLemmatizer()
            words = word_tokenize(text)
            text = ' '.join([lemmatizer.lemmatize(word) for word in words])
        except:
            pass
        
        return text
    
    def prepare_features(self):
        """Prepare features for training"""
        print("Preparing features...")
        
        # Text vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        text_features = self.vectorizer.fit_transform(self.clean_data['processed_text']).toarray()
        text_feature_names = self.vectorizer.get_feature_names_out()
        
        X_text = pd.DataFrame(text_features, columns=text_feature_names)
        
        # Add numerical features if available
        numerical_cols = []
        if 'telecommuting' in self.clean_data.columns:
            numerical_cols.append('telecommuting')
        if 'has_company_logo' in self.clean_data.columns:
            numerical_cols.append('has_company_logo')
        
        if numerical_cols:
            X_numerical = self.clean_data[numerical_cols].reset_index(drop=True)
            self.X = pd.concat([X_text, X_numerical], axis=1)
        else:
            self.X = X_text
        
        self.y = self.clean_data['fraudulent'].reset_index(drop=True)
        
        print(f"Feature matrix shape: {self.X.shape}")
        return self.X, self.y
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n=== TRAINING MULTIPLE MODELS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Handle class imbalance
        print("Applying ADASYN for class balancing...")
        adasyn = ADASYN(random_state=42)
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
        
        # Feature selection
        print("Performing feature selection...")
        self.selector = SelectFromModel(estimator=LinearSVC(random_state=42))
        X_train_selected = self.selector.fit_transform(X_train_balanced, y_train_balanced)
        X_test_selected = self.selector.transform(X_test)
        
        # Store for later use
        self.X_train_selected = X_train_selected
        self.X_test_selected = X_test_selected
        self.y_train = y_train_balanced
        self.y_test = y_test
        
        # Define models
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Linear SVM': LinearSVC(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'Passive Aggressive': PassiveAggressiveClassifier(random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train_selected, y_train_balanced)
            
            # Predictions
            y_pred_train = model.predict(X_train_selected)
            y_pred_test = model.predict(X_test_selected)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train_balanced, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test)
            
            training_time = time.time() - start_time
            
            # Store results
            self.model_performance[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
            }
            
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Select best model based on F1-score
        self.best_model_name = max(self.model_performance.keys(), 
                                 key=lambda k: self.model_performance[k]['f1_score'])
        self.best_model = self.model_performance[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name} (F1-Score: {self.model_performance[self.best_model_name]['f1_score']:.4f})")
        
        return self.model_performance
    
    def save_model_system(self):
        """Save the complete model system"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'selector': self.selector,
            'model_performance': {k: {
                'train_accuracy': v['train_accuracy'],
                'test_accuracy': v['test_accuracy'],
                'precision': v['precision'],
                'recall': v['recall'],
                'f1_score': v['f1_score'],
                'training_time': v['training_time'],
                'classification_report': v['classification_report']
            } for k, v in self.model_performance.items()},
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save model
        with open('fraud_detection_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save performance metrics as JSON
        with open('model_performance.json', 'w') as f:
            json.dump(model_data['model_performance'], f, indent=2)
        
        print("✅ Model system saved successfully!")
        print("📁 Files created:")
        print("   - fraud_detection_model.pkl (Complete model system)")
        print("   - model_performance.json (Performance metrics)")
    
    def predict_job(self, job_data):
        """Predict fraud probability for a single job"""
        if self.best_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create combined text
        combined_text = ' '.join([
            str(job_data.get('title', '')),
            str(job_data.get('location', '')),
            str(job_data.get('company_profile', '')),
            str(job_data.get('description', '')),
            str(job_data.get('requirements', '')),
            str(job_data.get('benefits', ''))
        ])
        
        # Preprocess
        processed_text = self._preprocess_text(combined_text)
        
        # Vectorize
        text_features = self.vectorizer.transform([processed_text]).toarray()
        
        # Add numerical features
        numerical_features = []
        if 'telecommuting' in job_data:
            numerical_features.append(job_data['telecommuting'])
        if 'has_company_logo' in job_data:
            numerical_features.append(job_data['has_company_logo'])
        
        # Combine features
        if numerical_features:
            features = np.concatenate([text_features[0], numerical_features]).reshape(1, -1)
        else:
            features = text_features
        
        # Select features
        features_selected = self.selector.transform(features)
        
        # Predict
        prediction = self.best_model.predict(features_selected)[0]
        probability = self.best_model.predict_proba(features_selected)[0, 1]
        
        return {
            'prediction': 'Fraudulent' if prediction == 1 else 'Genuine',
            'fraud_probability': float(probability),
            'confidence': float(max(self.best_model.predict_proba(features_selected)[0])),
            'model_used': self.best_model_name
        }
    
    def analyze_csv_dataset(self, csv_path):
        """Analyze a CSV dataset and return comprehensive results"""
        print(f"Analyzing CSV dataset: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} job postings")
        
        results = []
        
        for idx, row in df.iterrows():
            job_data = {
                'title': row.get('title', ''),
                'location': row.get('location', ''),
                'company_profile': row.get('company_profile', ''),
                'description': row.get('description', ''),
                'requirements': row.get('requirements', ''),
                'benefits': row.get('benefits', ''),
                'telecommuting': row.get('telecommuting', 0),
                'has_company_logo': row.get('has_company_logo', 0)
            }
            
            try:
                prediction_result = self.predict_job(job_data)
                
                results.append({
                    'id': idx + 1,
                    'title': job_data['title'],
                    'company': job_data.get('company_profile', 'Unknown')[:50],
                    'location': job_data['location'],
                    'prediction': prediction_result['prediction'],
                    'probability': prediction_result['fraud_probability'],
                    'confidence': prediction_result['confidence']
                })
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        total_jobs = len(results)
        fraudulent_jobs = len([r for r in results if r['prediction'] == 'Fraudulent'])
        genuine_jobs = total_jobs - fraudulent_jobs
        
        # Risk categorization
        low_risk = len([r for r in results if r['probability'] < 0.3])
        medium_risk = len([r for r in results if 0.3 <= r['probability'] < 0.7])
        high_risk = len([r for r in results if r['probability'] >= 0.7])
        
        # Calculate probability statistics
        probabilities = [r['probability'] for r in results]
        mean_prob = np.mean(probabilities) if probabilities else 0
        median_prob = np.median(probabilities) if probabilities else 0
        
        summary = {
            'total': total_jobs,
            'genuine': genuine_jobs,
            'fraudulent': fraudulent_jobs,
            'fraud_rate': (fraudulent_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'risk_breakdown': {
                'low_risk': low_risk,
                'medium_risk': medium_risk,
                'high_risk': high_risk
            },
            'probability_stats': {
                'mean': mean_prob,
                'median': median_prob,
                'min': min(probabilities) if probabilities else 0,
                'max': max(probabilities) if probabilities else 0
            }
        }
        
        return {
            'results': results,
            'summary': summary,
            'model_performance': self.model_performance
        }

def main():
    """Main training function"""
    print("🚀 Starting Comprehensive Fraud Detection Model Training")
    print("=" * 60)
    
    # Initialize detector
    detector = ComprehensiveFraudDetector()
    
    try:
        # Load and preprocess data
        detector.load_and_preprocess_data()
        
        # Prepare features
        detector.prepare_features()
        
        # Train models
        detector.train_models()
        
        # Save model system
        detector.save_model_system()
        
        print("\n" + "=" * 60)
        print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Display model comparison
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print("-" * 60)
        for name, metrics in detector.model_performance.items():
            print(f"{name}:")
            print(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Time:      {metrics['training_time']:.2f}s")
            print()
        
        print(f"🏆 Best Model: {detector.best_model_name}")
        print(f"🎯 Best F1-Score: {detector.model_performance[detector.best_model_name]['f1_score']:.4f}")
        
        # Test with example
        print("\n🔍 Testing with example job posting...")
        example_job = {
            'title': 'Software Engineer',
            'location': 'New York, NY',
            'company_profile': 'Leading tech company',
            'description': 'We are looking for a skilled software engineer to join our team...',
            'requirements': 'Bachelor degree in Computer Science, 3+ years experience',
            'benefits': 'Health insurance, 401k, flexible hours',
            'telecommuting': 1,
            'has_company_logo': 1
        }
        
        result = detector.predict_job(example_job)
        print(f"Prediction: {result['prediction']}")
        print(f"Fraud Probability: {result['fraud_probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Model Used: {result['model_used']}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
