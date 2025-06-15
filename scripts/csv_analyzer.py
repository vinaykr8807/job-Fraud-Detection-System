#!/usr/bin/env python3
"""
CSV Dataset Analyzer for Job Fraud Detection
This script analyzes uploaded CSV files and generates comprehensive reports
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_trained_model():
    """Load the pre-trained fraud detection model"""
    model_path = 'fraud_detection_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Please run enhanced_model_trainer.py first.")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def analyze_csv_file(csv_path, output_path='analysis_results.json'):
    """Analyze a CSV file and generate comprehensive results"""
    print(f"Loading trained model...")
    model_data = load_trained_model()
    
    best_model = model_data['best_model']
    vectorizer = model_data['vectorizer']
    selector = model_data['selector']
    
    print(f"Analyzing CSV file: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} job postings")
    
    results = []
    
    for idx, row in df.iterrows():
        # Extract job data
        job_data = {
            'title': str(row.get('title', '')),
            'location': str(row.get('location', '')),
            'company_profile': str(row.get('company_profile', '')),
            'description': str(row.get('description', '')),
            'requirements': str(row.get('requirements', '')),
            'benefits': str(row.get('benefits', ''))
        }
        
        try:
            # Create combined text
            combined_text = ' '.join(job_data.values())
            
            # Simple text preprocessing
            processed_text = combined_text.lower()
            processed_text = ' '.join(processed_text.split())
            
            # Vectorize
            text_features = vectorizer.transform([processed_text]).toarray()
            
            # Add numerical features if available
            numerical_features = []
            if 'telecommuting' in row:
                numerical_features.append(row.get('telecommuting', 0))
            if 'has_company_logo' in row:
                numerical_features.append(row.get('has_company_logo', 0))
            
            # Combine features
            if numerical_features:
                features = np.concatenate([text_features[0], numerical_features]).reshape(1, -1)
            else:
                features = text_features
            
            # Select features
            features_selected = selector.transform(features)
            
            # Predict
            prediction = best_model.predict(features_selected)[0]
            probability = best_model.predict_proba(features_selected)[0, 1]
            confidence = max(best_model.predict_proba(features_selected)[0])
            
            results.append({
                'id': idx + 1,
                'title': job_data['title'][:100],
                'company': job_data['company_profile'][:50] if job_data['company_profile'] else 'Unknown',
                'location': job_data['location'][:50],
                'prediction': 'Fraudulent' if prediction == 1 else 'Genuine',
                'probability': float(probability),
                'confidence': float(confidence)
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} jobs...")
    
    # Calculate comprehensive statistics
    total_jobs = len(results)
    fraudulent_jobs = len([r for r in results if r['prediction'] == 'Fraudulent'])
    genuine_jobs = total_jobs - fraudulent_jobs
    
    # Risk categorization
    low_risk = len([r for r in results if r['probability'] < 0.3])
    medium_risk = len([r for r in results if 0.3 <= r['probability'] < 0.7])
    high_risk = len([r for r in results if r['probability'] >= 0.7])
    
    # Probability statistics
    probabilities = [r['probability'] for r in results]
    mean_prob = np.mean(probabilities) if probabilities else 0
    median_prob = np.median(probabilities) if probabilities else 0
    
    # Create probability distribution for histogram
    prob_distribution = []
    for i in range(20):  # 20 bins for more detailed histogram
        start = i * 0.05
        end = (i + 1) * 0.05
        count = len([p for p in probabilities if start <= p < end])
        prob_distribution.append({
            'range': f"{start*100:.0f}-{end*100:.0f}%",
            'count': count,
            'percentage': (count / total_jobs * 100) if total_jobs > 0 else 0
        })
    
    # Summary statistics
    summary = {
        'total': total_jobs,
        'genuine': genuine_jobs,
        'fraudulent': fraudulent_jobs,
        'fraud_rate': (fraudulent_jobs / total_jobs * 100) if total_jobs > 0 else 0,
        'risk_breakdown': {
            'low_risk': low_risk,
            'low_risk_percentage': (low_risk / total_jobs * 100) if total_jobs > 0 else 0,
            'medium_risk': medium_risk,
            'medium_risk_percentage': (medium_risk / total_jobs * 100) if total_jobs > 0 else 0,
            'high_risk': high_risk,
            'high_risk_percentage': (high_risk / total_jobs * 100) if total_jobs > 0 else 0
        },
        'probability_stats': {
            'mean': float(mean_prob),
            'median': float(median_prob),
            'min': float(min(probabilities)) if probabilities else 0,
            'max': float(max(probabilities)) if probabilities else 0
        },
        'probability_distribution': prob_distribution
    }
    
    # Create final analysis result
    analysis_result = {
        'filename': os.path.basename(csv_path),
        'analysis_timestamp': datetime.now().isoformat(),
        'model_used': model_data['best_model_name'],
        'results': results,
        'summary': summary,
        'model_performance': model_data['model_performance']
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Results saved to: {output_path}")
    print(f"📈 Summary:")
    print(f"   Total Jobs: {total_jobs}")
    print(f"   Genuine: {genuine_jobs} ({genuine_jobs/total_jobs*100:.1f}%)")
    print(f"   Fraudulent: {fraudulent_jobs} ({fraudulent_jobs/total_jobs*100:.1f}%)")
    print(f"   Mean Fraud Probability: {mean_prob*100:.1f}%")
    
    return analysis_result

def main():
    """Main function for CSV analysis"""
    if len(sys.argv) != 2:
        print("Usage: python csv_analyzer.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        analyze_csv_file(csv_path)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
