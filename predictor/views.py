from django.shortcuts import render
from django.http import HttpResponse
import pickle
from PIL import Image
import numpy as np
from skimage.feature import hog
from django.core.files.storage import FileSystemStorage
import os

# Load model, scaler, and label encoder
with open('models/model_skin.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def home(request):
    return render(request, 'predictor/index.html')

def predict(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_path = fs.path(filename)
        
        try:
            # Load and preprocess image
            img = Image.open(uploaded_file_path).convert('RGB').resize((224, 224))
            img_arr = np.array(img)

            # Extract HOG features (same as training)
            fd = hog(
                img_arr, 
                orientations=9, 
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1
            )

            # Extract color histograms
            hist_r = np.histogram(img_arr[:, :, 0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_arr[:, :, 1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_arr[:, :, 2], bins=32, range=(0, 256))[0]

            # Combine features
            features = np.concatenate([fd, hist_r, hist_g, hist_b]).reshape(1, -1)

            # Apply scaling
            features_scaled = scaler.transform(features)

            # Predict
            pred = model.predict(features_scaled)
            label = le.inverse_transform(pred)[0]

        except Exception as e:
            os.remove(uploaded_file_path)
            return HttpResponse(f"Error processing image: {e}")

        # Clean up uploaded file
        os.remove(uploaded_file_path)
        
        return render(request, 'predictor/result.html', {'prediction': label})
        #return HttpResponse(f'Predicted skin defect: {label}')
    
    return HttpResponse("No file uploaded or invalid request.")
