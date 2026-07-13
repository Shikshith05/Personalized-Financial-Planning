"""
Prediction script for SMS classification using TensorFlow/Keras.
Allows users to input SMS and get predictions.
"""

import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_object
from preprocessing.preprocess import SMSPreprocessor
class SMSClassifier:
    """
    Wrapper class for SMS classification inference using TensorFlow/Keras.
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize classifier with model and preprocessor.
        
        Args:
            model_path: Path to trained model (uses default if None)
            preprocessor_path: Path to preprocessor (uses default if None)
        
        Raises:
            FileNotFoundError: If model or preprocessor files don't exist
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            model_path = os.path.join(script_dir, 'models', 'saved', 'best_model.keras')
        if preprocessor_path is None:
            preprocessor_path = os.path.join(script_dir, 'preprocessors', 'preprocessor.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        # Load preprocessor
        self.preprocessor: SMSPreprocessor = load_object(preprocessor_path)
        self.model = tf.keras.models.load_model(model_path)
        
        print("✓ Model and preprocessor loaded successfully")
    
    def predict(self, sms_text: str) -> Tuple[str, float]:
        """
        Predict class for input SMS.
        
        Args:
            sms_text: Input SMS text
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        try:
            # Preprocess
            X, _ = self.preprocessor.transform([sms_text])
            probs = self.model.predict(X, verbose=0)[0]

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100
            
            predicted_class = self.preprocessor.classes_[pred_idx]
            
            return predicted_class, confidence
        
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            raise
    
    def predict_batch(self, sms_texts: list) -> list:
        """
        Predict classes for multiple SMS.
        
        Args:
            sms_texts: List of SMS texts
        
        Returns:
            List of tuples (predicted_class, confidence)
        """
        results = []
        for sms in sms_texts:
            pred_class, confidence = self.predict(sms)
            results.append((pred_class, confidence))
        return results


def interactive_predict(model_path: str = None, preprocessor_path: str = None) -> None:
    """
    Interactive prediction loop.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
    """
    try:
        classifier = SMSClassifier(model_path, preprocessor_path)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nPlease ensure the model is trained first by running: python train.py")
        return
    
    print("\n" + "="*60)
    print("SMS CLASSIFICATION - INTERACTIVE PREDICTION (TensorFlow/Keras)")
    print("="*60)
    print("\nType an SMS to classify. Type 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            sms = input("Enter SMS: ").strip()
            
            if not sms:
                print("✗ Please enter a valid SMS\n")
                continue
            
            if sms.lower() in ['quit', 'exit']:
                print("\nThank you for using SMS Classifier!")
                break
            
            predicted_class, confidence = classifier.predict(sms)
            
            print(f"\nPredicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%\n")
        
        except KeyboardInterrupt:
            print("\n\nThank you for using SMS Classifier!")
            break
        except Exception as e:
            print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    interactive_predict()
