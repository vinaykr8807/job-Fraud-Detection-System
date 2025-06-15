#!/usr/bin/env python3
"""
Comprehensive Job Fraud Detection Model Trainer
Trains multiple models using synthetic data and evaluates performance
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Class balancing
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced models
import xgboost as xgb
try:
    from rgf.sklearn import RGFClassifier
    RGF_AVAILABLE = True
except ImportError:
    RGF_AVAILABLE = False
    print("Warning: RGF not available. Install with: pip install rgf_python")

warnings.filterwarnings("ignore")
plt.style.use('default')

class ComprehensiveFraudDetector:
    def __init__(self, synthetic_dataset_path="Training_Dataset.csv"):
        self.synthetic_dataset_path = synthetic_dataset_path
        self.models = {}
        self.vectorizer = None
        self.scaler = None
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Download NLTK data
        self._download_nltk_data()
        
        # Create output directories
        os.makedirs('model_outputs', exist_ok=True)
        os.makedirs('model_plots', exist_ok=True)
        
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        nltk_downloads = ['stopwords', 'punkt', 'wordnet', 'brown', 'punkt_tab']
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
            except:
                print(f"Warning: Could not download {item}")
    
    def load_synthetic_dataset(self):
        """Load the synthetic training dataset"""
        print("=" * 60)
        print("LOADING SYNTHETIC TRAINING DATASET")
        print("=" * 60)
        
        if not os.path.exists(self.synthetic_dataset_path):
            raise FileNotFoundError(f"Synthetic dataset not found: {self.synthetic_dataset_path}")
        
        # Load data
        self.raw_data = pd.read_csv(self.synthetic_dataset_path)
        print(f"✅ Dataset loaded: {self.raw_data.shape}")
        
        # Display dataset info
        print(f"📊 Dataset Overview:")
        print(f"   Total samples: {len(self.raw_data):,}")
        print(f"   Features: {len(self.raw_data.columns)}")
        print(f"   Missing values: {self.raw_data.isnull().sum().sum()}")
        
        # Class distribution
        fraud_counts = self.raw_data['fraudulent'].value_counts()
        print(f"\n🎯 Class Distribution:")
        print(f"   Genuine jobs: {fraud_counts[0]:,} ({fraud_counts[0]/len(self.raw_data)*100:.2f}%)")
        print(f"   Fraudulent jobs: {fraud_counts[1]:,} ({fraud_counts[1]/len(self.raw_data)*100:.2f}%)")
        
        return self.raw_data
    
    def preprocess_data(self):
        """Comprehensive data preprocessing pipeline"""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Start with a copy of raw data
        self.clean_data = self.raw_data.copy()
        
        # Remove duplicates
        initial_shape = self.clean_data.shape[0]
        self.clean_data = self.clean_data.drop_duplicates()
        print(f"🧹 Removed {initial_shape - self.clean_data.shape[0]:,} duplicate rows")
        
        # Handle missing values for text columns
        text_columns = ['title', 'location', 'department', 'company_profile', 
                       'description', 'requirements', 'benefits', 'employment_type',
                       'required_experience', 'required_education', 'industry', 'function']
        
        for col in text_columns:
            if col in self.clean_data.columns:
                self.clean_data[col] = self.clean_data[col].fillna("")
        
        # Create combined text feature
        print("📝 Creating combined text features...")
        self.clean_data['combined_text'] = (
            self.clean_data.get('title', '').astype(str) + ' ' +
            self.clean_data.get('location', '').astype(str) + ' ' +
            self.clean_data.get('company_profile', '').astype(str) + ' ' +
            self.clean_data.get('description', '').astype(str) + ' ' +
            self.clean_data.get('requirements', '').astype(str) + ' ' +
            self.clean_data.get('benefits', '').astype(str)
        )
        
        # Text preprocessing
        print("🔤 Preprocessing text data...")
        self.clean_data['processed_text'] = self.clean_data['combined_text'].apply(self._preprocess_text)
        
        # Calculate text features
        self.clean_data['text_length'] = self.clean_data['processed_text'].apply(len)
        self.clean_data['word_count'] = self.clean_data['processed_text'].apply(lambda x: len(x.split()))
        
        # Handle numerical features
        numerical_features = ['telecommuting', 'has_company_logo']
        for feature in numerical_features:
            if feature in self.clean_data.columns:
                self.clean_data[feature] = self.clean_data[feature].fillna(0)
        
        print(f"✅ Preprocessed dataset shape: {self.clean_data.shape}")
        return self.clean_data
    
    def _preprocess_text(self, text):
        """Advanced text preprocessing pipeline"""
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
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n" + "=" * 60)
        print("FEATURE PREPARATION")
        print("=" * 60)
        
        # Text vectorization
        print("🔢 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.95,
            stop_words='english'
        )
        
        text_features = self.vectorizer.fit_transform(self.clean_data['processed_text']).toarray()
        text_feature_names = self.vectorizer.get_feature_names_out()
        
        X_text = pd.DataFrame(text_features, columns=text_feature_names)
        
        # Add numerical features
        numerical_cols = ['text_length', 'word_count']
        if 'telecommuting' in self.clean_data.columns:
            numerical_cols.append('telecommuting')
        if 'has_company_logo' in self.clean_data.columns:
            numerical_cols.append('has_company_logo')
        
        X_numerical = self.clean_data[numerical_cols].reset_index(drop=True)
        
        # Combine features
        self.X = pd.concat([X_text, X_numerical], axis=1)
        self.y = self.clean_data['fraudulent'].reset_index(drop=True)
        
        # Store feature names
        self.feature_names = list(self.X.columns)
        
        print(f"✅ Feature matrix shape: {self.X.shape}")
        print(f"📊 Text features: {len(text_feature_names):,}")
        print(f"📊 Numerical features: {len(numerical_cols)}")
        
        return self.X, self.y
    
    def train_multiple_models(self, test_size=0.2, random_state=42):
        """Train and evaluate multiple machine learning models"""
        print("\n" + "=" * 60)
        print("TRAINING MULTIPLE MODELS")
        print("=" * 60)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Define models with SMOTE integration
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'use_smote': True
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'use_smote': True
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=random_state,
                    gamma='scale'
                ),
                'use_smote': True
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'use_smote': True
            }
        }
        
        # Add RGF if available
        if RGF_AVAILABLE:
            models_config['RGF'] = {
                'model': RGFClassifier(
                    max_leaf=1000,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=False
                ),
                'use_smote': True
            }
        
        # Train and evaluate each model
        for name, config in models_config.items():
            print(f"\n🚀 Training {name}...")
            start_time = time.time()
            
            try:
                # Create pipeline with SMOTE
                if config['use_smote']:
                    # Create SMOTE pipeline
                    smote = SMOTE(random_state=random_state, k_neighbors=3)
                    pipeline = ImbPipeline([
                        ('smote', smote),
                        ('classifier', config['model'])
                    ])
                else:
                    pipeline = config['model']
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                precision = precision_score(y_test, y_pred_test)
                recall = recall_score(y_test, y_pred_test)
                f1 = f1_score(y_test, y_pred_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
                
                training_time = time.time() - start_time
                
                # Generate classification report
                class_report = classification_report(y_test, y_pred_test, output_dict=True)
                
                # Store results
                self.model_performance[name] = {
                    'model': pipeline,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'classification_report': class_report,
                    'y_pred': y_pred_test,
                    'y_pred_proba': y_pred_proba
                }
                
                # Display results
                print(f"   ✅ Training completed in {training_time:.2f}s")
                print(f"   📊 Test Accuracy: {test_accuracy:.4f}")
                print(f"   📊 Precision: {precision:.4f}")
                print(f"   📊 Recall: {recall:.4f}")
                print(f"   📊 F1-Score: {f1:.4f}")
                print(f"   📊 ROC-AUC: {roc_auc:.4f}")
                print(f"   📊 CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                continue
        
        # Select best model based on F1-score
        if self.model_performance:
            self.best_model_name = max(self.model_performance.keys(), 
                                     key=lambda k: self.model_performance[k]['f1_score'])
            self.best_model = self.model_performance[self.best_model_name]['model']
            
            print(f"\n🏆 Best model: {self.best_model_name}")
            print(f"🎯 Best F1-Score: {self.model_performance[self.best_model_name]['f1_score']:.4f}")
        
        return self.model_performance
    
    def display_detailed_results(self):
        """Display comprehensive results for all models"""
        print("\n" + "=" * 80)
        print("DETAILED MODEL PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, metrics in self.model_performance.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'CV F1': f"{metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})",
                'Time (s)': f"{metrics['training_time']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 MODEL COMPARISON SUMMARY:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        
        # Detailed classification reports
        print("\n📋 DETAILED CLASSIFICATION REPORTS:")
        print("=" * 80)
        
        for name, metrics in self.model_performance.items():
            print(f"\n🔍 {name.upper()} - CLASSIFICATION REPORT:")
            print("-" * 50)
            print(classification_report(self.y_test, metrics['y_pred']))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            print(f"\n📊 {name.upper()} - CONFUSION MATRIX:")
            print(f"                 Predicted")
            print(f"                 0      1")
            print(f"Actual    0   {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"          1   {cm[1,0]:4d}   {cm[1,1]:4d}")
            
            # Additional metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\n📈 {name.upper()} - ADDITIONAL METRICS:")
            print(f"   Sensitivity (Recall): {sensitivity:.4f}")
            print(f"   Specificity: {specificity:.4f}")
            print(f"   False Positive Rate: {fp/(fp+tn):.4f}")
            print(f"   False Negative Rate: {fn/(fn+tp):.4f}")
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        print("\n" + "=" * 60)
        print("CREATING PERFORMANCE VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Model Comparison Chart
        self._plot_model_comparison()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Precision-Recall Curves
        self._plot_precision_recall_curves()
        
        # 4. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 5. Feature Importance (for tree-based models)
        self._plot_feature_importance()
        
        print("✅ All visualizations saved to 'model_plots/' directory")
    
    def _plot_model_comparison(self):
        """Plot comparison of different models"""
        metrics = ['test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.model_performance.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.model_performance[name][metric] for name in model_names]
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            
            bars = axes[i].bar(model_names, values, color=colors)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=12)
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig('model_plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.model_performance)))
        
        for i, (name, metrics) in enumerate(self.model_performance.items()):
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            auc_score = metrics['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves Comparison', fontweight='bold', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.model_performance)))
        
        for i, (name, metrics) in enumerate(self.model_performance.items()):
            precision, recall, _ = precision_recall_curve(self.y_test, metrics['y_pred_proba'])
            
            plt.plot(recall, precision, color=colors[i], lw=2, label=f'{name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curves Comparison', fontweight='bold', fontsize=16)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.model_performance)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, (name, metrics) in enumerate(self.model_performance.items()):
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotations
            annotations = []
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    annotations.append(f'{cm[row,col]}\n({cm_percent[row,col]:.1f}%)')
            
            annotations = np.array(annotations).reshape(cm.shape)
            
            sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                       xticklabels=['Genuine', 'Fraudulent'],
                       yticklabels=['Genuine', 'Fraudulent'],
                       ax=axes[i])
            axes[i].set_title(f'{name}', fontweight='bold')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            axes[j].remove()
        
        plt.tight_layout()
        plt.savefig('model_plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'XGBoost']
        available_tree_models = [name for name in tree_models if name in self.model_performance]
        
        if not available_tree_models:
            return
        
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(15, 8))
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_tree_models):
            model = self.model_performance[model_name]['model']
            
            # Extract the actual classifier from pipeline
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps['classifier']
            else:
                classifier = model
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get top 20 features
                top_indices = np.argsort(importances)[-20:]
                top_features = [self.feature_names[idx] for idx in top_indices]
                top_importances = importances[top_indices]
                
                # Create plot
                axes[i].barh(range(len(top_features)), top_importances)
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features)
                axes[i].set_xlabel('Feature Importance')
                axes[i].set_title(f'{model_name} - Top 20 Features')
        
        plt.tight_layout()
        plt.savefig('model_plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_complete_model_system(self):
        """Save the complete model system"""
        print("\n" + "=" * 60)
        print("SAVING MODEL SYSTEM")
        print("=" * 60)
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'all_models': {name: metrics['model'] for name, metrics in self.model_performance.items()},
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'model_performance': {k: {
                'train_accuracy': v['train_accuracy'],
                'test_accuracy': v['test_accuracy'],
                'precision': v['precision'],
                'recall': v['recall'],
                'f1_score': v['f1_score'],
                'roc_auc': v['roc_auc'],
                'cv_mean': v['cv_mean'],
                'cv_std': v['cv_std'],
                'training_time': v['training_time'],
                'classification_report': v['classification_report']
            } for k, v in self.model_performance.items()},
            'training_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.clean_data),
                'fraud_rate': len(self.clean_data[self.clean_data['fraudulent'] == 1]) / len(self.clean_data),
                'features_count': len(self.feature_names)
            }
        }
        
        # Save complete model system
        with open('model_outputs/fraud_detection_model_system.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save performance metrics as JSON
        with open('model_outputs/model_performance_detailed.json', 'w') as f:
            json.dump(model_data['model_performance'], f, indent=2)
        
        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'best_f1_score': self.model_performance[self.best_model_name]['f1_score'],
            'dataset_size': len(self.clean_data),
            'models_trained': list(self.model_performance.keys()),
            'feature_count': len(self.feature_names)
        }
        
        with open('model_outputs/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Complete model system saved!")
        print("📁 Files created:")
        print("   - model_outputs/fraud_detection_model_system.pkl")
        print("   - model_outputs/model_performance_detailed.json")
        print("   - model_outputs/training_summary.json")

def main():
    """Main training function"""
    print("🚀 COMPREHENSIVE FRAUD DETECTION MODEL TRAINING")
    print("=" * 80)
    print("This system trains multiple ML models on synthetic data only.")
    print("User-uploaded datasets will be used only for predictions.")
    print("=" * 80)
    
    # Initialize detector
    detector = ComprehensiveFraudDetector()
    
    try:
        # Check if synthetic dataset exists
        if not os.path.exists("Training_Dataset.csv"):
            print("❌ Synthetic training dataset not found!")
            print("📝 Please run: python scripts/synthetic_dataset_generator.py")
            return
        
        # Load synthetic dataset
        detector.load_synthetic_dataset()
        
        # Preprocess data
        detector.preprocess_data()
        
        # Prepare features
        detector.prepare_features()
        
        # Train multiple models
        detector.train_multiple_models()
        
        # Display detailed results
        detector.display_detailed_results()
        
        # Create visualizations
        detector.create_performance_visualizations()
        
        # Save complete system
        detector.save_complete_model_system()
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"🏆 Best Model: {detector.best_model_name}")
        print(f"🎯 Best F1-Score: {detector.model_performance[detector.best_model_name]['f1_score']:.4f}")
        print(f"📊 Models Trained: {len(detector.model_performance)}")
        print(f"📈 Visualizations: Saved to 'model_plots/' directory")
        print(f"💾 Model Files: Saved to 'model_outputs/' directory")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
