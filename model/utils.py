import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from itertools import cycle

class ModelEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training history - simplified version"""
        
        # Check if history is a History object or our custom object
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history
        
        # Determine which metrics are available
        available_metrics = list(hist.keys())
        has_precision = 'precision' in available_metrics
        has_recall = 'recall' in available_metrics
        
        if has_precision and has_recall:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
        epochs = range(1, len(hist['accuracy']) + 1)
        
        if has_precision and has_recall:
            # Full plot with all metrics
            # Accuracy
            axes[0, 0].plot(epochs, hist['accuracy'], 'bo-', label='Training', linewidth=2)
            axes[0, 0].plot(epochs, hist['val_accuracy'], 'ro-', label='Validation', linewidth=2)
            axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss
            axes[0, 1].plot(epochs, hist['loss'], 'bo-', label='Training', linewidth=2)
            axes[0, 1].plot(epochs, hist['val_loss'], 'ro-', label='Validation', linewidth=2)
            axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Precision
            axes[1, 0].plot(epochs, hist['precision'], 'go-', label='Training', linewidth=2)
            axes[1, 0].plot(epochs, hist['val_precision'], 'mo-', label='Validation', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Recall
            axes[1, 1].plot(epochs, hist['recall'], 'co-', label='Training', linewidth=2)
            axes[1, 1].plot(epochs, hist['val_recall'], 'yo-', label='Validation', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Simplified plot with just accuracy and loss
            # Accuracy
            axes[0].plot(epochs, hist['accuracy'], 'bo-', label='Training', linewidth=2)
            axes[0].plot(epochs, hist['val_accuracy'], 'ro-', label='Validation', linewidth=2)
            axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss
            axes[1].plot(epochs, hist['loss'], 'bo-', label='Training', linewidth=2)
            axes[1].plot(epochs, hist['val_loss'], 'ro-', label='Validation', linewidth=2)
            axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot confusion matrix with percentages"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations with counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row_annotations = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j] * 100
                annotation = f'{count}\n({percentage:.1f}%)'
                row_annotations.append(annotation)
            annotations.append(row_annotations)
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def combine_histories(history1, history2):
    """Combine two training histories"""
    combined = {}
    for key in history1.history.keys():
        if key in history2.history:
            combined[key] = history1.history[key] + history2.history[key]
        else:
            combined[key] = history1.history[key]
    
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return CombinedHistory(combined)