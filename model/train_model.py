import numpy as np
import tensorflow as tf
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coral_model import CoralHealthModel
from utils import ModelEvaluator, combine_histories

def load_processed_data(data_path='data/patches/processed_data.pkl'):
    """Load processed data from pickle file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Please run data_processor.py first.")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def create_data_splits(patches, labels, test_size=0.2, val_size=0.2, random_state=42):
    """Create train/validation/test splits"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Create label encoder
    health_classes = ['healthy', 'bleached', 'dead']
    label_encoder = LabelEncoder()
    label_encoder.fit(health_classes)
    
    # Encode labels
    labels_encoded = label_encoder.transform(labels)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        patches, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def create_simple_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """Create simple data generators without complex augmentation"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Normalize data
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    # Simple augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    return train_generator, val_generator

def main():
    """Main training function"""
    print("="*80)
    print("CORAL REEF HEALTH DETECTION - MODEL TRAINING")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load processed data
    print("Loading processed data...")
    try:
        data = load_processed_data()
        patches = data['patches']
        labels = data['labels']
        health_classes = data['health_classes']
        patch_size = data.get('patch_size', 64)
        
        print(f"✓ Loaded {len(patches):,} patches")
        print(f"✓ Patch size: {patch_size}x{patch_size}")
        print(f"✓ Classes: {health_classes}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run 'python data_processor.py' first to process the dataset.")
        return
    
    # Create data splits
    print("\nCreating data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = create_data_splits(patches, labels)
    
    print(f"✓ Train set: {len(X_train):,} samples")
    print(f"✓ Validation set: {len(X_val):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")
    
    # Show class distribution
    from collections import Counter
    print(f"\nClass distribution in training set:")
    train_dist = Counter(y_train)
    for class_idx, count in train_dist.items():
        class_name = label_encoder.inverse_transform([class_idx])[0]
        print(f"  {class_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    # Create model
    print("\nCreating model...")
    input_shape = (patch_size, patch_size, 3)
    model_builder = CoralHealthModel(input_shape=input_shape, num_classes=len(health_classes))
    
    # Use custom CNN model instead of transfer learning to avoid metric issues
    print("Building Custom CNN model...")
    model = model_builder.create_custom_cnn_model()
    
    # Compile with simpler metrics to avoid batch size issues
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Only use accuracy to avoid shape issues
    )
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Create simple data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_simple_data_generators(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Setup callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/coral_health_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    try:
        history = model.fit(
            train_gen,
            epochs=50,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Falling back to simple training without generators...")
        
        # Fallback: Simple training without generators
        X_train_norm = X_train.astype('float32') / 255.0
        X_val_norm = X_val.astype('float32') / 255.0
        
        history = model.fit(
            X_train_norm, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(X_val_norm, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Create evaluator
    evaluator = ModelEvaluator(health_classes)
    
    # Plot training history
    print("\nPlotting training history...")
    evaluator.plot_training_history(history, 'training_history.png')
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = 'models/coral_health_model.h5'
    if os.path.exists(best_model_path):
        best_model = tf.keras.models.load_model(best_model_path)
        print(f"✓ Loaded best model from {best_model_path}")
    else:
        best_model = model
        print("Using current model (best model file not found)")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Simple evaluation
    X_test_norm = X_test.astype('float32') / 255.0
    test_loss, test_accuracy = best_model.evaluate(X_test_norm, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions for detailed evaluation
    predictions = best_model.predict(X_test_norm, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_test, predicted_classes, 
                                 target_names=health_classes, output_dict=True)
    
    print("\nDetailed Classification Report:")
    for class_name in health_classes:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {int(metrics['support'])}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    evaluator.plot_confusion_matrix(cm, 'confusion_matrix.png')
    
    # Save label encoder
    print("\nSaving label encoder...")
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"✓ Best model saved: {best_model_path}")
    print(f"✓ Label encoder saved: models/label_encoder.pkl")
    print(f"✓ Test accuracy: {test_accuracy:.4f}")
    
    if 'macro avg' in report:
        print(f"✓ Test macro F1: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\nPer-class performance:")
    for class_name in health_classes:
        if class_name in report:
            f1 = report[class_name]['f1-score']
            print(f"  {class_name}: F1 = {f1:.4f}")
    
    print(f"\nGenerated files:")
    print(f"  • models/coral_health_model.h5 (trained model)")
    print(f"  • models/label_encoder.pkl (label encoder)")
    print(f"  • training_history.png (training curves)")
    print(f"  • confusion_matrix.png (evaluation results)")
    
    print(f"\nNext step: Run the web application with: python app.py")

if __name__ == "__main__":
    main()