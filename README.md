# MRI Breast Tumor Classification Model & Flask Server

This project is a Flask backend server for image classification using a pretrained ResNet50-based deep learning model.  
Users can upload an image, and the server returns a classification label (**Malignant** or **Benign**) along with a confidence score.

---

## 🧠 Features

- Accepts image uploads via `/predict` endpoint  
- Preprocesses images with ResNet50 preprocessing  
- Returns prediction and confidence in JSON format  
- Cross-Origin Resource Sharing (CORS) enabled for frontend integration

---

## ⚙️ Requirements

- Python 3.8+  
- TensorFlow 2.x  
- Flask  
- Flask-CORS  
- Pillow (PIL)  
- NumPy

Install them with:

```bash
pip install -r requirements.txt
```
---
## ⚙️ References :
- Rapport : [DOCS LINK](https://docs.google.com/document/d/1d1XFF3STLU7m9i3RSE8XJDk011o8tbl0FfrOKNpLKyU/edit?usp=sharing)<br>
- Dataset Kaggle Cleaned and Splitted : [Kaggle Dataset](https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset)<br>
- Source of MRIs : [NCIA Archive](https://www.cancerimagingarchive.net/collection/breast-diagnosis/)<br>




