# Skin Cancer Classification & Segmentation | HAM10000

## About
This project is part of **SifAI**, a dual-model AI system for medical image analysis.  
It focuses on **skin cancer detection and lesion segmentation**, processing 10,000+ dermoscopic images.  
You can see the other model from skin cancer module from here: [GitHub](https://github.com/gumaruw/Skin-Cancer-Binary). 

## Datasets
- Skin Cancer MNIST HAM10000 (Images): [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- HAM10000 Lesion Segmentations (Masks): [Kaggle](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations/data)
  
<!-- ## Model 
- https://drive.google.com/drive/folders/1RKIYV1J6zSe0zLmM7GaBnvOKj0qh-77-?usp=sharing -->

## Project Structure
project/  
├─ data/           — HAM10000 dataset    
│   ├─ images/     — Training & validation images    
│   └─ masks/      — Corresponding segmentation masks    
├─ model/          — Saved/produced model files    
├─ test_results/   — Output predictions & metrics    
├─ venv/           — Python virtual environment    
├─ requirements.txt     
├─ .gitignore         
├─ train.py        — Training script      
├─ train_no_lambda_layer.py — Training variant without lambda layer      
└─ test_unet.py    — U-Net testing script               

---

## Model & Performance
### Segmentation (U-Net)
- **Accuracy:** 95.83%  
- **IoU:** 84.95%  
- **Dice Coefficient:** 91.83%  

**Fast test on 50 images:**  
- Mean IoU: 0.8315 ± 0.1518  
- Mean Dice: 0.8986 ± 0.1141  
- Min IoU: 0.3051 | Max IoU: 0.9706  

## Strengths
- High segmentation accuracy with 90%+ Dice coefficient  
- Excellent overlap for most lesions (IoU > 0.8)  
- Robust classification suitable for clinical use  

## Considerations
- Lower performance on 10–15% of difficult lesions  
- Some variation in IoU indicates challenging cases  

## Usage
1. Load images and masks from HAM10000.  
2. Train or load pre-trained U-Net for segmentation.  
3. Train or load ResNet50 for malignant vs. benign classification.  
4. Evaluate using IoU, Dice, and accuracy metrics.  

## Live Demo
A video showcasing **model integration into the SifAI platform** is here:  [Google Drive Folder](https://drive.google.com/drive/folders/14x163_HpD7DB1LjPwVKphoIj2z6kkeHR?usp=sharing)

## Screenshots From SifAI
<img width="800" height="393" alt="image" src="https://github.com/user-attachments/assets/d2369b8c-6b3b-4c70-98ce-bee6eb8fbf50" />
<img width="800" height="382" alt="image" src="https://github.com/user-attachments/assets/7438a2a5-8605-40ed-9a9a-a57b9c7d4bc0" />
<img width="800" height="386" alt="image" src="https://github.com/user-attachments/assets/0eeac84f-340a-4246-b780-f4236c46b440" />)
<img width="800" height="363" alt="image" src="https://github.com/user-attachments/assets/b4d3c60d-7494-4b9b-bf16-454057ba8e7d" />
<img width="800" height="385" alt="image" src="https://github.com/user-attachments/assets/440aa743-1826-4b32-bd17-dc49c4409942" />

## Notes
- Designed for research and educational purposes.
- Model can be further improved by increasing dataset size or fine-tuning hyperparameters.

## Achievements
- Developed dual-model AI system for **skin cancer detection and brain tumor analysis**  
- Built U-Net segmentation model with **95.83% pixel accuracy** and **91.83% Dice coefficient**  
- Trained ResNet50 classifier achieving **89.55% accuracy** and **93.88% AUC**  
- Processed 10,000+ dermoscopic images with advanced augmentation and custom loss functions  
- Delivered a complete deep learning pipeline using **TensorFlow, Keras, and OpenCV** 
- Collaborated in a 5-person team integrating skin cancer module with brain tumor classifier for thesis project

## Usage:
```bash
git clone https://github.com/gumaruw/Skin-Cancer-SifAI.git
```
```bash
cd Skin-Cancer-SifAI
```
```bash
pip install -r requirements.txt
```
