# Image Classification Project
## Project Overview
    Classify_Image is a Python-based image classification system that categorizes images into three classes: human, animal, and plant using machine learning techniques.

## Project Goal
    To build an efficient image classifier that distinguishes between different biological categories using feature extraction and decision tree classification.

## Tech Stack & Tools

### Programming Language
![python](https://img.shields.io/badge/Python-3.x-blue?logo=python)

### Core Libraries
![sklearn](https://img.shields.io/badge/scikit--learn-0.24+-orange?logo=scikit-learn)
![opencv](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![numpy](https://img.shields.io/badge/NumPy-1.19+-blue?logo=numpy)

## Required installations
    pip install -f requirements.txt

📁 Project Structure
```bash
Classify_Image/
├── src/
|   ├── main.py
|   └── classify_image.py
├── datasets/
│   ├── human/
│   ├── animal/
│   └── plant/
├── requirements.txt
└── README.md
```
## Class Architecture
### Class: Classify_Image
    A machine learning classifier that processes images and predicts their category.

### Class Variables
1. __output: Maps categories to numerical labels

2. "human" → 0

3. "animal" → 1

4. "plant" → 2

5. NEW_SIZE_IMAGE: Standardized image size (128×128 pixels)

### Constructor: __init__()
```python
def __init__(self, folder_human_path, folder_animal_path, folder_planet_path)
```
### Parameters:

1. folder_human_path: Path to human images directory

2. folder_animal_path: Path to animal images directory

3. folder_planet_path: Path to plant images directory

### Process:

- Initializes Decision Tree classifier

- Loads and processes images from all three directories

- Extracts features and builds training dataset

### Private Methods
```python
__extract_color(image)
```
#### Purpose: Extracts color histogram features from images

### Process:

- Resizes image to standard size

- For RGB images: Extracts histograms for each channel (R, G, B)

- For grayscale images: Extracts single histogram

- Returns flattened histogram features

```python
__extract_edge(image)
```
#### Purpose: Extracts edge features using Canny edge detection

### Process:

- Resizes image to standard size

- Applies Canny edge detection (thresholds: 50, 200)

- Returns flattened edge features

```python
__main_process(image)
```
### Purpose: Main preprocessing pipeline for training images

## Process:

- Resizes input image

- Converts to grayscale for edge detection

- Extracts both color and edge features

- Concatenates features and adds to training set

## Public Methods
#### fit_model()
```python
def fit_model(self)
```
### Purpose: Trains the Decision Tree classifier on extracted features

## Process:

- Fits the model using stored __x_train and __y_train data

#### real_time(path_image)
```python
def real_time(self, path_image)
```
### Purpose: Predicts category for a single image

## Parameters:

1. path_image: Path to the image file

## Returns:

- Predicted class label (0, 1, or 2)

## Process:

- Reads and resizes image

- Extracts features (edges + color)

- Makes prediction using trained model

#### test_accuracy(paths, y_test)
```python
def test_accuracy(self, paths, y_test)
```
### Purpose: Evaluates model accuracy on test dataset

## Parameters:

1. paths: List of image paths for testing

2. y_test: True labels for test images

## Returns:

- Accuracy score (float between 0 and 1)

##  Usage Example
```python
# Initialize classifier with training data
classifier = Classify_Image(
    folder_human_path='images/human',
    folder_animal_path='images/animal',
    folder_planet_path='images/plant'
)

# Train the model
classifier.fit_model()

# Make a prediction
prediction = classifier.real_time('test_image.jpg')
print(f"Predicted class: {prediction}")

# Test accuracy
test_paths = ['test1.jpg', 'test2.jpg', 'test3.jpg']
true_labels = [0, 1, 2]  # human, animal, plant
accuracy = classifier.test_accuracy(test_paths, true_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```
## Methodology
### Feature Extraction Pipeline
- Image Standardization: All images resized to 128×128 pixels

- Color Features: RGB/grayscale histograms (256 bins per channel)

- Texture Features: Canny edge detection for shape information

- Feature Fusion: Concatenation of color and edge features

### Classification Model
- Algorithm: Decision Tree Classifier

#### Advantages:

- Fast training and prediction

- Interpretable decision rules

- No feature scaling required

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/yourusername/image-classification.git
cd image-classification

# Install dependencies
pip install -r requirements.txt

# Organize your dataset
mkdir -p images/human images/animal images/plant

# Add images to respective folders
# images/human/*.jpg
# images/animal/*.jpg  
# images/plant/*.jpg
```

## Requirements File

```bash
scikit-learn>=0.24.2
opencv-python>=4.5.3
numpy>=1.19.5
```

## Limitations & Considerations
- Image Requirements:

- Supports JPG, PNG formats

- Color images (RGB) or grayscale

- Minimum size: 128×128 pixels

## Performance Factors:

- Decision trees may overfit with small datasets

- Feature extraction is computationally intensive

- Accuracy depends on image quality and diversity

## Scalability:

- Suitable for small to medium datasets

- For larger datasets, consider CNNs or transfer learning

## Future Enhancements
### Model Improvements:

- Implement Random Forest for better accuracy

- Add CNN-based feature extraction

- Include data augmentation techniques

### Features:

- Real-time webcam classification

- Batch prediction for multiple images

- Export trained model for deployment

### Interface:

- Web interface using Flask/Django

- REST API for remote classification

- Mobile application integration