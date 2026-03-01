"""
model_utils.py
Model architecture builders for RetinaScan AI

Contains:
- EfficientNetB3 with transfer learning
- ResNet50 with transfer learning
- Ensemble model wrapper
- Model utilities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50
import numpy as np


def build_efficientnet(
    input_shape=(224, 224, 3),
    num_classes=5,
    weights='imagenet',
    freeze_base=True,
    dropout_rate=0.5
):
    """
    EfficientNetB3 model with transfer learning
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        freeze_base: Whether to freeze base model layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB3
    base_model = EfficientNetB3(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='avg'  # Global average pooling
    )
    
    # Freeze base model if specified
    base_model.trainable = not freeze_base
    
    # Build model
    model = models.Sequential([
        base_model,
        
        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='EfficientNetB3_DR')
    
    return model


def build_resnet50(
    input_shape=(224, 224, 3),
    num_classes=5,
    weights='imagenet',
    freeze_base=True,
    dropout_rate=0.5
):
    """
    ResNet50 model with transfer learning
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        freeze_base: Whether to freeze base model layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='avg'  # Global average pooling
    )
    
    # Freeze base model if specified
    base_model.trainable = not freeze_base
    
    # Build model
    model = models.Sequential([
        base_model,
        
        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='ResNet50_DR')
    
    return model


def unfreeze_top_layers(model, num_layers=20):
    """
    Unfreeze top N layers of base model for fine-tuning
    
    Args:
        model: Keras model
        num_layers: Number of top layers to unfreeze
        
    Returns:
        Modified model with unfrozen layers
    """
    # Get the base model (first layer)
    base_model = model.layers[0]
    
    # Freeze all layers first
    base_model.trainable = True
    
    # Freeze all but the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    print(f"✓ Unfroze top {num_layers} layers for fine-tuning")
    print(f"  Trainable params: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    return model


def compile_model(
    model,
    learning_rate=0.001,
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=None
):
    """
    Compile model with specified optimizer and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        loss: Loss function
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    
    # Select optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Compile
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_callbacks(
    model_name,
    save_dir='models',
    monitor='val_accuracy',
    patience=7,
    reduce_lr_patience=3,
    min_lr=1e-7
):
    """
    Creates standard callbacks for training
    
    Args:
        model_name: Name for saving model checkpoints
        save_dir: Directory to save models
        monitor: Metric to monitor for callbacks
        patience: Patience for early stopping
        reduce_lr_patience: Patience for reducing learning rate
        min_lr: Minimum learning rate
        
    Returns:
        List of callbacks
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks_list = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=f'{save_dir}/{model_name}_best.h5',
            monitor=monitor,
            save_best_only=True,
            mode='max' if 'accuracy' in monitor else 'min',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            f'{save_dir}/{model_name}_training_log.csv',
            separator=',',
            append=False
        )
    ]
    
    return callbacks_list


class EnsembleModel:
    """
    Ensemble wrapper for combining multiple models
    """
    
    def __init__(self, model_paths, weights=None):
        """
        Initialize ensemble with multiple models
        
        Args:
            model_paths: List of paths to saved models
            weights: Optional list of weights for each model (must sum to 1)
        """
        self.models = []
        self.model_paths = model_paths
        
        # Load all models
        print("Loading models for ensemble...")
        for i, path in enumerate(model_paths):
            try:
                model = keras.models.load_model(path)
                self.models.append(model)
                print(f"  ✓ Loaded model {i+1}: {path}")
            except Exception as e:
                print(f"  ✗ Error loading {path}: {e}")
        
        # Set weights
        if weights is None:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            assert len(weights) == len(self.models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = np.array(weights)
        
        print(f"\n✓ Ensemble ready with {len(self.models)} models")
        print(f"  Weights: {self.weights}")
    
    def predict(self, X, method='weighted_average'):
        """
        Makes ensemble prediction
        
        Args:
            X: Input data (single image or batch)
            method: Ensemble method ('weighted_average', 'voting', 'max')
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Combine predictions
        if method == 'weighted_average':
            # Weighted average of probabilities
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += self.weights[i] * pred
        
        elif method == 'voting':
            # Majority voting (convert to class labels first)
            class_preds = [np.argmax(pred, axis=1) for pred in predictions]
            class_preds = np.array(class_preds).T
            ensemble_pred = np.array([
                np.bincount(row).argmax() for row in class_preds
            ])
        
        elif method == 'max':
            # Maximum probability across models
            ensemble_pred = np.maximum.reduce(predictions)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred
    
    def evaluate(self, X, y, method='weighted_average'):
        """
        Evaluates ensemble on test data
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            method: Ensemble method
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        # Get predictions
        ensemble_pred = self.predict(X, method=method)
        
        # Convert to class labels
        if len(ensemble_pred.shape) > 1:
            y_pred = np.argmax(ensemble_pred, axis=1)
        else:
            y_pred = ensemble_pred
        
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE EVALUATION (Method: {method})")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(
            y_true, 
            y_pred, 
            target_names=[f'Grade {i}' for i in range(5)]
        ))
        
        return {
            'accuracy': accuracy,
            'predictions': ensemble_pred,
            'y_true': y_true,
            'y_pred': y_pred
        }


def get_model_summary(model, print_summary=True):
    """
    Gets comprehensive model summary
    
    Args:
        model: Keras model
        print_summary: Whether to print the summary
        
    Returns:
        Dictionary with model information
    """
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    non_trainable_params = total_params - trainable_params
    
    info = {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    if print_summary:
        print(f"\n{'='*60}")
        print(f"MODEL SUMMARY: {info['name']}")
        print(f"{'='*60}")
        print(f"Total parameters:        {info['total_params']:>15,}")
        print(f"Trainable parameters:    {info['trainable_params']:>15,}")
        print(f"Non-trainable parameters:{info['non_trainable_params']:>15,}")
        print(f"Number of layers:        {info['num_layers']:>15}")
        print(f"Input shape:             {str(info['input_shape']):>15}")
        print(f"Output shape:            {str(info['output_shape']):>15}")
        print(f"{'='*60}\n")
    
    return info


def save_model_architecture(model, filepath):
    """
    Save model architecture as JSON
    
    Args:
        model: Keras model
        filepath: Path to save JSON file
    """
    import json
    
    architecture = model.to_json()
    
    with open(filepath, 'w') as f:
        json.dump(json.loads(architecture), f, indent=2)
    
    print(f"✓ Model architecture saved to: {filepath}")


def load_model_from_architecture(architecture_path, weights_path=None):
    """
    Load model from architecture JSON and optionally load weights
    
    Args:
        architecture_path: Path to architecture JSON file
        weights_path: Optional path to weights file
        
    Returns:
        Keras model
    """
    import json
    
    with open(architecture_path, 'r') as f:
        architecture = json.load(f)
    
    model = keras.models.model_from_json(json.dumps(architecture))
    
    if weights_path:
        model.load_weights(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
    
    return model


def main():
    """
    Test model builders
    """
    print("\n" + "="*60)
    print("TESTING MODEL BUILDERS")
    print("="*60)
    
    # Test EfficientNet
    print("\n1. Building EfficientNetB3...")
    efficientnet = build_efficientnet(
        input_shape=(224, 224, 3),
        num_classes=5,
        weights=None,  # No pre-trained weights for testing
        freeze_base=True
    )
    efficientnet = compile_model(efficientnet, learning_rate=0.001)
    get_model_summary(efficientnet)
    
    # Test ResNet50
    print("\n2. Building ResNet50...")
    resnet = build_resnet50(
        input_shape=(224, 224, 3),
        num_classes=5,
        weights=None,
        freeze_base=True
    )
    resnet = compile_model(resnet, learning_rate=0.001)
    get_model_summary(resnet)
    
    print("\n" + "="*60)
    print(" MODEL BUILDERS WORKING CORRECTLY!")
    print("="*60)


if __name__ == "__main__":
    main()