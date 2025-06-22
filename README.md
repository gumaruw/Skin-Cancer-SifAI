# Skin-Cancer-Classification-Segmentation-HAM10000 

## Dataset İndirme
- Skin Cancer MNIST HAM10000 - Ana görüntüler: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- HAM10000 Lesion Segmentations - Segmentasyon maskeleri: https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations/data

## Model 
- https://drive.google.com/drive/folders/1RKIYV1J6zSe0zLmM7GaBnvOKj0qh-77-?usp=sharing

---

## Test Results

- **Accuracy:** 95.83% — Very high pixel-level accuracy
- **IoU:** 84.95% — Excellent segmentation overlap
- **Dice Coefficient:** 91.83% — Outstanding score for medical segmentation

### Fast Test Results on 50 Images

- **Mean IoU:** 0.8315 ± 0.1518
- **Mean Dice:** 0.8986 ± 0.1141
- **Min IoU:** 0.3051
- **Max IoU:** 0.9706

---

## Performance Analysis

### ✅ Strengths

- **Mean IoU: 0.8315** — Excellent for medical segmentation
- **Mean Dice: 0.8986** — Close to 90%, a great result
- **Max IoU: 0.9706** — Near-perfect segmentation on some images

### ⚠️ Points to Consider

- **Min IoU: 0.3051** — Lower performance on some difficult cases
- **IoU Std: 0.1518** — Slightly high variation, indicates some inconsistency

---

## Evaluation

These results are at a level accepted in the literature for skin cancer lesion segmentation. In particular:

- **IoU > 0.8** → Very successful segmentation
- **Dice > 0.89** → Sufficient sensitivity for clinical use
- Only **10-15%** of images show low performance

---

## Conclusion

The model performs very well on **85–90%** of images, only struggling with very difficult or ambiguous lesions. These are truly successful results for medical image processing projects!
