import os
import numpy as np
import cv2
import pickle
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
label_encoder = None
patch_size = 64

# Class descriptions
class_descriptions = {
    'healthy': 'Coral appears to be in good health with normal structure, vibrant colors, and active polyps. This indicates a thriving ecosystem.',
    'bleached': 'Coral shows signs of stress, potentially bleached due to temperature changes or affected by algae growth. Requires monitoring and intervention.',
    'dead': 'Coral appears to be dead or severely damaged with broken structure. This represents significant ecosystem loss requiring immediate conservation action.'
}

# Health recommendations
health_recommendations = {
    'healthy': [
        "Continue regular monitoring of water quality parameters",
        "Maintain stable water temperature (26-29°C optimal)",
        "Monitor for early signs of disease or bleaching",
        "Reduce human disturbance and boat anchoring in the area",
        "Document healthy coral locations for conservation planning",
        "Implement protective measures against crown-of-thorns starfish"
    ],
    'bleached': [
        "Immediate assessment of water temperature and quality",
        "Reduce additional stressors (pollution, sedimentation, overfishing)",
        "Monitor closely for signs of recovery or further deterioration",
        "Consider temporary shading or cooling interventions if possible",
        "Increase monitoring frequency to twice weekly",
        "Restrict tourism and diving activities in affected areas",
        "Test for coral diseases that may accompany bleaching"
    ],
    'dead': [
        "Document the extent and potential causes of coral mortality",
        "Remove dead coral carefully to prevent disease spread",
        "Assess and address underlying environmental factors",
        "Consider active coral restoration and replanting efforts",
        "Prevent further degradation of surrounding healthy coral",
        "Implement immediate protective measures for nearby reefs",
        "Conduct water quality testing for pollutants and toxins"
    ]
}

def load_coral_model():
    """Load the trained coral health model and label encoder"""
    global model, label_encoder
    
    try:
        # Try to load the trained model
        model_paths = [
            'models/coral_health_model.h5',
            'models/best_coral_model.h5'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                model = load_model(model_path)
                model_loaded = True
                break
        
        if not model_loaded:
            print("No trained model found. Creating dummy model...")
            create_dummy_model()
            return
        
        # Load label encoder
        encoder_path = 'models/label_encoder.pkl'
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("Label encoder loaded successfully!")
        else:
            print("Label encoder not found. Creating dummy encoder...")
            create_dummy_encoder()
        
        print("Model and encoder loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model classes: {label_encoder.classes_}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        create_dummy_model()

def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    global model, label_encoder
    from sklearn.preprocessing import LabelEncoder
    
    print("Creating dummy model for demonstration...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    create_dummy_encoder()

def create_dummy_encoder():
    """Create a dummy label encoder"""
    global label_encoder
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['healthy', 'bleached', 'dead'])

def extract_patches_from_image(image, grid_size=12):
    """Extract patches from image for analysis using grid sampling"""
    h, w = image.shape[:2]
    
    patches = []
    patch_positions = []
    
    # Calculate step sizes for grid
    step_h = h // grid_size
    step_w = w // grid_size
    half_patch = patch_size // 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate center position
            center_y = int((i + 0.5) * step_h)
            center_x = int((j + 0.5) * step_w)
            
            # Check if patch fits within image bounds
            if (center_y - half_patch >= 0 and center_y + half_patch < h and
                center_x - half_patch >= 0 and center_x + half_patch < w):
                
                # Extract patch
                y1, y2 = center_y - half_patch, center_y + half_patch
                x1, x2 = center_x - half_patch, center_x + half_patch
                
                patch = image[y1:y2, x1:x2]
                
                if patch.shape[:2] == (patch_size, patch_size):
                    patches.append(patch)
                    patch_positions.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    })
    
    return np.array(patches), patch_positions

def analyze_coral_health(image_path):
    """Analyze coral health in the uploaded image"""
    try:
        print(f"Analyzing image: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image.shape}")
        
        # Extract patches using grid sampling
        patches, positions = extract_patches_from_image(image, grid_size=12)
        print(f"Extracted {len(patches)} patches")
        
        if len(patches) == 0:
            print("Error: No patches could be extracted")
            return None
        
        # Normalize patches
        patches_norm = patches.astype('float32') / 255.0
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(patches_norm, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
        confidences = np.max(predictions, axis=1)
        
        # Calculate overall health statistics
        health_counts = {class_name: 0 for class_name in label_encoder.classes_}
        confidence_sums = {class_name: 0.0 for class_name in label_encoder.classes_}
        
        for label, conf in zip(predicted_labels, confidences):
            health_counts[label] += 1
            confidence_sums[label] += float(conf)
        
        total_patches = len(patches)
        health_percentages = {label: (count / total_patches) * 100 
                            for label, count in health_counts.items()}
        
        # Calculate average confidence per class
        avg_confidences = {}
        for label in label_encoder.classes_:
            if health_counts[label] > 0:
                avg_confidences[label] = confidence_sums[label] / health_counts[label]
            else:
                avg_confidences[label] = 0.0
        
        # Determine overall health status (weighted by confidence)
        weighted_scores = {label: health_percentages[label] * avg_confidences[label] 
                          for label in label_encoder.classes_}
        dominant_class = max(weighted_scores, key=weighted_scores.get)
        overall_confidence = avg_confidences[dominant_class]
        
        # Calculate health score (0-100, higher is better)
        health_score = (
            health_percentages.get('healthy', 0) * 1.0 +
            health_percentages.get('bleached', 0) * 0.5 +
            health_percentages.get('dead', 0) * 0.0
        )
        
        # Create detailed patch results
        patch_results = []
        for i, (pos, pred_label, conf) in enumerate(zip(positions, predicted_labels, confidences)):
            patch_results.append({
                'patch_id': i,
                'position': pos,
                'prediction': pred_label,
                'confidence': float(conf),
                'color': get_health_color(pred_label)
            })
        
        # Prepare comprehensive results
        results = {
            'overall_health': str(dominant_class), 
            'overall_confidence': float(overall_confidence),
            'health_score': float(health_score),
            'health_percentages': health_percentages,
            'avg_confidences': avg_confidences,
            'patch_predictions': patch_results,
            'total_patches_analyzed': int(total_patches),
            'image_dimensions': {
                'width': int(image.shape[1]), 
                'height': int(image.shape[0])
            },
            'description': class_descriptions[dominant_class],
            'analysis_summary': generate_analysis_summary(health_percentages, health_score)
        }
        
        print(f"Analysis complete. Overall health: {dominant_class} ({overall_confidence:.2f} confidence)")
        return results
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_health_color(health_status):
    """Get color code for health status"""
    colors = {
        'healthy': '#2ecc71',    # Green
        'bleached': '#f39c12',   # Orange  
        'dead': '#e74c3c'        # Red
    }
    return colors.get(health_status, '#95a5a6')  # Gray for unknown

def generate_analysis_summary(health_percentages, health_score):
    """Generate a text summary of the analysis"""
    healthy_pct = health_percentages.get('healthy', 0)
    bleached_pct = health_percentages.get('bleached', 0)
    dead_pct = health_percentages.get('dead', 0)
    
    if health_score >= 75:
        condition = "excellent"
    elif health_score >= 50:
        condition = "good"
    elif health_score >= 25:
        condition = "concerning"
    else:
        condition = "critical"
    
    summary = f"Reef condition is {condition} with a health score of {health_score:.1f}/100. "
    summary += f"Analysis shows {healthy_pct:.1f}% healthy coral, {bleached_pct:.1f}% bleached/stressed areas, "
    summary += f"and {dead_pct:.1f}% dead coral. "
    
    if healthy_pct > 60:
        summary += "The reef shows strong resilience and biodiversity."
    elif bleached_pct > 40:
        summary += "Significant bleaching detected - immediate monitoring recommended."
    elif dead_pct > 30:
        summary += "High mortality rate detected - urgent conservation action needed."
    
    return summary

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(np.random.random() * 1000000))
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"File saved to: {file_path}")
            
            # Analyze coral health
            results = analyze_coral_health(file_path)
            
            if results is None:
                return jsonify({'error': 'Error analyzing image. Please try a different image.'}), 400
            
            # Add image path and recommendations
            results['image_path'] = f'/uploads/{filename}'
            results['recommendations'] = health_recommendations.get(results['overall_health'], [])
            
            return jsonify(results)
        
        return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG image.'}), 400
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error during analysis: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None,
        'classes': list(label_encoder.classes_) if label_encoder else None
    })

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'classes': list(label_encoder.classes_) if label_encoder else None,
            'patch_size': patch_size
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("CORAL REEF HEALTH DETECTION WEB APPLICATION")
    print("="*60)
    print("Loading coral reef health detection model...")
    load_coral_model()
    print("Starting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)