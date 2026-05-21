"""
Evaluation script for model performance assessment.
"""

import os
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             auc, classification_report)
from pathlib import Path

from data_processing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, preprocessor_path: str, data_path: str, 
                  output_dir: str):
    """
    Evaluate trained model on test data.
    
    Args:
        model_path: Path to trained model pickle file
        preprocessor_path: Path to preprocessor pickle file
        data_path: Path to test/eval data
        output_dir: Directory to save evaluation results
    """
    logger.info("Starting model evaluation")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and preprocessor
        logger.info("Loading model and preprocessor...")
        model = joblib.load(model_path)
        processor = DataProcessor()
        processor.load_preprocessor(preprocessor_path)
        
        # Process evaluation data
        logger.info("Processing evaluation data...")
        X_eval = processor.process_data(data_path, is_training=False)
        
        # Load raw data to get actual labels
        df = processor.load_data(data_path)
        y_eval = df['Survived'].values
        
        logger.info(f"Evaluation set size: {X_eval.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = {
            'accuracy': float(accuracy_score(y_eval, y_pred)),
            'precision': float(precision_score(y_eval, y_pred, zero_division=0)),
            'recall': float(recall_score(y_eval, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_eval, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_eval, y_pred_proba))
        }
        
        cm = confusion_matrix(y_eval, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_eval, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {roc_path}")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns_imported = False
        try:
            import seaborn as sns
            sns_imported = True
        except ImportError:
            pass
        
        if sns_imported:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Did not survive', 'Survived'],
                       yticklabels=['Did not survive', 'Survived'])
        else:
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Did not survive', 'Survived'])
            plt.yticks(tick_marks, ['Did not survive', 'Survived'])
            
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.grid(False)
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        # Print classification report
        logger.info("\nDetailed Classification Report:")
        print(classification_report(y_eval, y_pred,
                                   target_names=['Did not survive', 'Survived']))
        
        logger.info("Evaluation completed successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='../model/model.pkl',
                       help='Path to trained model')
    parser.add_argument('--preprocessor', type=str, default='../model/preprocessor.pkl',
                       help='Path to preprocessor')
    parser.add_argument('--data', type=str, default='../titanic.csv',
                       help='Path to evaluation data')
    parser.add_argument('--output', type=str, default='../evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    model_path = os.path.abspath(args.model)
    preprocessor_path = os.path.abspath(args.preprocessor)
    data_path = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output)
    
    evaluate_model(model_path, preprocessor_path, data_path, output_dir)


if __name__ == '__main__':
    main()
