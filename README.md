# Skin-Cancer-Classification-Segmentation-HAM10000 

## Dataset İndirme
- Skin Cancer MNIST HAM10000 - Ana görüntüler: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- HAM10000 Lesion Segmentations - Segmentasyon maskeleri: https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations/data

## Klasör Yapısı
unet_skin_project/
├── model/
│   └── unet_lesion_model.h5          # Modeliniz
├── data/
│   ├── images/                       # Ana görüntüler
│   │   ├── HAM10000_images_part_1/
│   │   │   ├── ISIC_0024306.jpg
│   │   │   ├── ISIC_0024307.jpg
│   │   │   └── ... (7000+ dosya)
│   │   └── HAM10000_images_part_2/
│   │       ├── ISIC_0034321.jpg
│   │       └── ... (3000+ dosya)
│   └── masks/                        # Segmentasyon maskeleri
│       └── HAM10000_segmentations_lesion_tschandl/
│           ├── ISIC_0024306_segmentation.png
│           ├── ISIC_0024307_segmentation.png
│           └── ... (10000+ dosya)
├── test_results/                     # Sonuçlar için
├── test_unet.py                      # Test kodu
└── requirements.txt

## Vs-Code 
- Virtual environment oluştur
python -m venv unet_env

- Aktive et (Windows)
unet_env\Scripts\activate

- Kütüphaneleri yükle
pip install tensorflow matplotlib opencv-python pillow numpy

- Test kodunu kullan
- Virtual Environment:

