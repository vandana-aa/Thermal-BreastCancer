# 🔬 Thermal Breast Cancer Detection

A full-stack web application for breast cancer screening using thermal imaging. Built with **Flask (backend)** and **React (frontend)**, it uses deep learning models for segmentation and classification of thermal breast images.

---

## 🌐 Live Demo

- 🔸 Frontend (React): [https://thermal-breast-cancer.vercel.app](https://thermal-breast-cancer.vercel.app)  
- 🔸 Backend (Flask): [https://thermal-breastcancer-production.up.railway.app](https://thermal-breastcancer-production.up.railway.app)

---

## 🧠 Features

- Upload thermal breast images
- Segment breast region using a UNet model (`.h5`)
- Split left and right breast regions
- Classify each side as **Benign** or **Malignant** using a ConvNeXt-Tiny model (`.pth`)
- View all processed images (segmented, left, right, original)
- Confidence scores displayed alongside results

---

## 🖼 Workflow Overview

1. User uploads a thermal breast image via the frontend.
2. Image is sent to the Flask backend for processing.
3. Backend:
   - Segments breast region using a UNet model.
   - Splits left and right regions.
   - Classifies both using a ConvNeXt-Tiny model from `timm`.
4. JSON response contains:
   - Predicted labels (`Benign` / `Malignant`)
   - Confidence scores
   - URLs for segmented and classified images
5. Frontend displays diagnosis + visual output.

---

## 🛠 Technologies Used

### Frontend
- React
- TailwindCSS
- Vercel (Deployment)

### Backend
- Flask
- Flask-CORS
- PyTorch
- Keras / TensorFlow
- `timm` (ConvNeXt)
- Gunicorn (for production server)
- Railway (Deployment)

---

## 🗂 Project Structure


---

## 🚀 Deployment Instructions

### Backend (Flask) – Railway
1. Push backend to GitHub
2. Create Railway project → "Deploy from GitHub"
3. Add a `Procfile`:

web: gunicorn app:app


4. Upload `.h5` and `.pth` model files via Railway shell:
```bash
mkdir -p models
wget <your-model-url> -O models/unet_model.h5
wget <your-model-url> -O models/convnext_tiny.fb_in22k_vmodel.pth

### Frontend (React) – Vercel
1. Push React app to GitHub

2. Create Vercel project → Connect GitHub
3. Add environment variable:
REACT_APP_API_URL=https://thermal-breastcancer-production.up.railway.app
4. Build and deploy



