import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class CoralHealthModel:
    def __init__(self, input_shape=(64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_custom_cnn_model(self):
        """Create custom CNN model optimized for coral patch classification"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def create_transfer_learning_model(self, base_model_name='EfficientNetB0'):
        """Create transfer learning model using pre-trained networks"""
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def fine_tune_model(self, base_layers_to_unfreeze=30):
        """Fine-tune the pre-trained base model"""
        if self.model is None:
            raise ValueError("Model must be created first")
            
        # Get the base model (first layer)
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze bottom layers, unfreeze top layers
        total_layers = len(base_model.layers)
        for layer in base_model.layers[:-base_layers_to_unfreeze]:
            layer.trainable = False
            
        print(f"Fine-tuning: Unfroze top {base_layers_to_unfreeze} layers out of {total_layers}")
        
        # Use much lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self, save_path='models/best_coral_model.h5'):
        """Get training callbacks for optimal training"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks