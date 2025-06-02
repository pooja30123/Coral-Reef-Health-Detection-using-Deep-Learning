# 🌊 Coral Reef Health Detection 

This project is a **group project** developed by students of **IIIT Lucknow**. It is a Flask-based deep learning web application that detects the health status of coral reefs from underwater images. The system uses a trained CNN model to classify reef images into categories like *healthy*, *bleached*, or *diseased*.

---

## 📌 Features

- Upload coral reef images via a user-friendly web interface  
- Predict reef health using a trained deep learning model  
- Displays classification result and confidence score  
- Save and view uploaded images for future reference

---

## 🧠 Model Details

- **Architecture**: CNN (Convolutional Neural Network)  
- **Training Data**: Custom dataset of labeled coral reef images  
- **Input Size**: 224x224 RGB images  
- **Framework**: TensorFlow / Keras  

---

## 💻 Tech Stack

| Component        | Technology        |
|------------------|-------------------|
| Backend          | Python, Flask     |
| Deep Learning    | TensorFlow / Keras|
| Frontend         | HTML, CSS (Jinja) |
| Visualization    | Matplotlib, Seaborn|
| Deployment Ready | Flask App         |

--- 

## 🗂️ Project Structure

```
├── data/
│ ├── patches/
│ │ └── processed_data.pkl
│ ├── processed/
│ │ └── coral-reef-dataset.zip
│
├── model/
│ ├── init.py
│ ├── coral_model.py
│ ├── point_classifier.py
│ ├── train_model.py
│ ├── utils.py
│
├── models/
│ ├── coral_health_model.h5
│ ├── label_encoder.pkl
│
├── static/
│ ├── css/
│ ├── js/
│ └── uploads/
│
├── templates/
│ └── index.html
│
├── app.py # Flask application script
├── data_processor.py # For preprocessing and patch handling
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── confusion_matrix.png # Evaluation output
├── dataset_analysis.png # Dataset overview
├── sample_patches.png # Sample input patches
├── training_history.png # Training curves
```

---

## 🚀 How to Run the App Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/coral-reef-health-detection.git
cd coral-reef-health-detection
```
### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```
### 3. Install the dependencies
```
pip install -r requirements.txt
```
### 4. Run the Flask app
```
python app.py
````
---