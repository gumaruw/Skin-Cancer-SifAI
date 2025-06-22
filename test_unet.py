# test_unet.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import glob
from pathlib import Path

# Model yükleme - Lambda layer problemi için düzeltilmiş
def load_model_safe(model_path):
    """Güvenli model yükleme - Lambda layer problemini çözer"""
    
    # Custom metrics
    def pixel_accuracy(y_true, y_pred):
        correct_pixels = tf.equal(y_true, tf.round(y_pred))
        return tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    
    def iou(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(tf.round(y_pred), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        return intersection / (union + tf.keras.backend.epsilon())
    
    def dice_coefficient(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(tf.round(y_pred), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 2. * intersection / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())
    
    # Lambda layer için özel fonksiyon
    def normalize_layer(x):
        return x / 255.0
    
    custom_objects = {
        'pixel_accuracy': pixel_accuracy,
        'iou': iou,
        'dice_coefficient': dice_coefficient,
        '<lambda>': normalize_layer,  # Lambda layer için
        'lambda': normalize_layer,
        'normalize_layer': normalize_layer
    }
    
    try:
        # İlk deneme - standart yükleme
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ Model başarıyla yüklendi!")
        return model
    except Exception as e:
        print(f"⚠️ Standart yükleme başarısız: {e}")
        
        try:
            # İkinci deneme - compile=False ile yükleme
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model compile=False ile yüklendi!")
            
            # Modeli yeniden compile et
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=[pixel_accuracy, iou, dice_coefficient]
            )
            print("✅ Model yeniden compile edildi!")
            return model
            
        except Exception as e2:
            print(f"❌ Model yükleme başarısız: {e2}")
            print("\n🔧 Çözüm önerileri:")
            print("1. Modeli SavedModel formatında kaydedin")
            print("2. Lambda layer'ı düzenli fonksiyon ile değiştirin")
            print("3. Model eğitimini tekrar yapın")
            return None

# Dosya eşleştirme
def find_matching_files(images_dir, masks_dir):
    """Görüntü ve mask dosyalarını eşleştir"""
    
    # Tüm görüntü dosyalarını bul
    image_patterns = [
        os.path.join(images_dir, "HAM10000_images_part_1", "*.jpg"),
        os.path.join(images_dir, "HAM10000_images_part_2", "*.jpg")
    ]
    
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(pattern))
    
    # Mask dosyalarını bul
    mask_pattern = os.path.join(masks_dir, "HAM10000_segmentations_lesion_tschandl", "*.png")
    all_masks = glob.glob(mask_pattern)
    
    # Eşleştirme
    matched_pairs = []
    
    for img_path in all_images:
        img_name = Path(img_path).stem  # Dosya adı (uzantısız)
        
        # Corresponding mask dosyasını ara
        mask_name = f"{img_name}_segmentation.png"
        mask_path = os.path.join(masks_dir, "HAM10000_segmentations_lesion_tschandl", mask_name)
        
        if os.path.exists(mask_path):
            matched_pairs.append((img_path, mask_path))
    
    print(f"📊 {len(all_images)} görüntü, {len(all_masks)} mask bulundu")
    print(f"🔗 {len(matched_pairs)} eşleşen çift bulundu")
    
    return matched_pairs

# Görüntü preprocessing
def preprocess_image(image_path):
    """Görüntüyü model için hazırla"""
    image = Image.open(image_path).convert('RGB')
    original = image.copy()
    
    # 256x256'ya resize
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Manuel normalizasyon
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array, original

def preprocess_mask(mask_path):
    """Mask'i yükle ve hazırla"""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((256, 256))
    mask_array = np.array(mask) / 255.0
    
    return mask_array

# Ana test fonksiyonu
def comprehensive_skin_test(model, image_path, mask_path, threshold=0.5, save_result=False):
    """Kapsamlı cilt lezyonu testi"""
    
    # Dosya adını al
    filename = Path(image_path).stem
    
    print(f"🔍 Test ediliyor: {filename}")
    
    # Görüntü ve mask'i yükle
    processed_img, original_img = preprocess_image(image_path)
    ground_truth = preprocess_mask(mask_path)
    
    # AI Prediction
    try:
        ai_prediction = model.predict(processed_img, verbose=0)[0]
        ai_binary = (ai_prediction > threshold).astype(np.uint8)
        
        # Metrikleri hesapla
        gt_binary = (ground_truth > 0.5).astype(int)
        pred_binary = ai_binary.squeeze().astype(int)
        
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        iou = intersection / union if union > 0 else 0
        
        dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary)) if (np.sum(pred_binary) + np.sum(gt_binary)) > 0 else 0
        
        accuracy = np.mean(pred_binary == gt_binary)
        
        # Overlay görüntüleri oluştur
        original_resized = np.array(Image.open(image_path).convert('RGB').resize((256, 256)))
        
        # Ground truth overlay (yeşil)
        gt_overlay = original_resized.copy()
        gt_mask = ground_truth > 0.5
        gt_overlay[gt_mask] = [0, 255, 0]  # Yeşil
        gt_combined = cv2.addWeighted(original_resized, 0.7, gt_overlay, 0.3, 0)
        
        # AI prediction overlay (mavi)
        ai_overlay = original_resized.copy()
        ai_mask = ai_binary.squeeze() > 0.5
        ai_overlay[ai_mask] = [0, 0, 255]  # Mavi
        ai_combined = cv2.addWeighted(original_resized, 0.7, ai_overlay, 0.3, 0)
        
        # Görselleştirme (2x3 grid)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Skin Lesion Segmentation Analysis - {filename}\nIoU: {iou:.3f} | Dice: {dice:.3f} | Accuracy: {accuracy:.3f}', 
                     fontsize=14, fontweight='bold')
        
        # Üst sıra
        axes[0,0].imshow(original_resized)
        axes[0,0].set_title('Original Skin Image', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(ground_truth, cmap='gray')
        axes[0,1].set_title('Ground Truth Mask', fontweight='bold')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(ai_binary.squeeze(), cmap='gray')
        axes[0,2].set_title('AI Predicted Mask', fontweight='bold')
        axes[0,2].axis('off')
        
        # Alt sıra - Overlays
        axes[1,0].imshow(gt_combined)
        axes[1,0].set_title('Image + Ground Truth', fontweight='bold', color='green')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(ai_combined)
        axes[1,1].set_title('Image + AI Prediction', fontweight='bold', color='blue')
        axes[1,1].axis('off')
        
        # Probability heatmap
        axes[1,2].imshow(ai_prediction.squeeze(), cmap='hot')
        axes[1,2].set_title('Prediction Confidence', fontweight='bold')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        
        if save_result:
            save_path = f'test_results/{filename}_analysis.png'
            os.makedirs('test_results', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Sonuç kaydedildi: {save_path}")
        
        plt.show()
        
        return {
            'filename': filename,
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'prediction': ai_prediction,
            'binary_mask': ai_binary
        }
        
    except Exception as e:
        print(f"❌ Prediction hatası: {e}")
        return None

# Batch test fonksiyonu
def batch_skin_test(model, matched_pairs, num_samples=5, save_results=True):
    """Birden fazla örneği test et"""
    
    if not matched_pairs:
        print("❌ Test edilecek görüntü bulunamadı!")
        return []
    
    # Rastgele örnekler seç
    num_samples = min(num_samples, len(matched_pairs))
    selected_pairs = np.random.choice(len(matched_pairs), num_samples, replace=False)
    
    results = []
    
    print(f"🧪 {num_samples} örnek test ediliyor...\n")
    
    for i, idx in enumerate(selected_pairs):
        img_path, mask_path = matched_pairs[idx]
        print(f"Test {i+1}/{num_samples}: {Path(img_path).name}")
        
        result = comprehensive_skin_test(model, img_path, mask_path, save_result=save_results)
        if result:
            results.append(result)
            print(f"   IoU: {result['iou']:.3f} | Dice: {result['dice']:.3f} | Acc: {result['accuracy']:.3f}\n")
        else:
            print("   ❌ Test başarısız!\n")
    
    if results:
        # Ortalama metrikleri hesapla
        avg_iou = np.mean([r['iou'] for r in results])
        avg_dice = np.mean([r['dice'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        
        print("="*50)
        print("📈 BATCH TEST SONUÇLARI:")
        print(f"Ortalama IoU: {avg_iou:.3f}")
        print(f"Ortalama Dice: {avg_dice:.3f}")
        print(f"Ortalama Accuracy: {avg_acc:.3f}")
        print("="*50)
    
    return results

# Farklı threshold'ları karşılaştır
def threshold_comparison(model, image_path, mask_path, thresholds=[0.3, 0.5, 0.7, 0.9]):
    """Farklı threshold değerlerini karşılaştır"""
    
    processed_img, _ = preprocess_image(image_path)
    ground_truth = preprocess_mask(mask_path)
    
    try:
        prediction = model.predict(processed_img, verbose=0)[0]
        
        fig, axes = plt.subplots(2, len(thresholds), figsize=(16, 8))
        fig.suptitle(f'Threshold Comparison - {Path(image_path).name}', fontsize=14, fontweight='bold')
        
        metrics = []
        
        for i, threshold in enumerate(thresholds):
            binary_pred = (prediction > threshold).astype(int)
            gt_binary = (ground_truth > 0.5).astype(int)
            
            # Metrikleri hesapla
            intersection = np.sum(binary_pred.squeeze() * gt_binary)
            union = np.sum(binary_pred.squeeze()) + np.sum(gt_binary) - intersection
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (np.sum(binary_pred.squeeze()) + np.sum(gt_binary)) if (np.sum(binary_pred.squeeze()) + np.sum(gt_binary)) > 0 else 0
            
            metrics.append({'threshold': threshold, 'iou': iou, 'dice': dice})
            
            # Görselleştir
            axes[0,i].imshow(binary_pred.squeeze(), cmap='gray')
            axes[0,i].set_title(f'Threshold: {threshold}')
            axes[0,i].axis('off')
            
            axes[1,i].text(0.5, 0.7, f'IoU: {iou:.3f}', ha='center', va='center', 
                          transform=axes[1,i].transAxes, fontsize=12, fontweight='bold')
            axes[1,i].text(0.5, 0.3, f'Dice: {dice:.3f}', ha='center', va='center', 
                          transform=axes[1,i].transAxes, fontsize=12, fontweight='bold')
            axes[1,i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Threshold comparison hatası: {e}")
        return []

# Ana çalıştırma fonksiyonu
def main():
    """Ana program"""
    
    # Yolları tanımla
    MODEL_PATH = "model/unet_lesion_model.h5"
    IMAGES_DIR = "data/images"
    MASKS_DIR = "data/masks"
    
    print("🩺 Cilt Lezyonu Segmentasyon Testi")
    print("="*50)
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        print("Model dosyasının doğru konumda olduğundan emin olun.")
        return
    
    # Model yükle
    model = load_model_safe(MODEL_PATH)
    
    if model is None:
        print("❌ Model yüklenemedi! Program sonlandırılıyor.")
        return
    
    # Dosyaları eşleştir
    matched_pairs = find_matching_files(IMAGES_DIR, MASKS_DIR)
    
    if not matched_pairs:
        print("❌ Eşleşen dosya bulunamadı! Klasör yapısını kontrol edin.")
        print(f"Beklenen yapı:")
        print(f"  {IMAGES_DIR}/HAM10000_images_part_1/*.jpg")
        print(f"  {IMAGES_DIR}/HAM10000_images_part_2/*.jpg")
        print(f"  {MASKS_DIR}/HAM10000_segmentations_lesion_tschandl/*.png")
        return
    
    print("\n🎯 Test seçenekleri:")
    print("1. Tek örnek test")
    print("2. Batch test (5 örnek)")
    print("3. Threshold karşılaştırması")
    print("4. Rastgele 10 örnek")
    print("5. Model bilgilerini göster")
    
    choice = input("\nSeçiminiz (1-5): ")
    
    if choice == "1":
        # Tek örnek
        img_path, mask_path = matched_pairs[0]
        comprehensive_skin_test(model, img_path, mask_path, save_result=True)
    
    elif choice == "2":
        # Batch test
        batch_skin_test(model, matched_pairs, num_samples=5)
    
    elif choice == "3":
        # Threshold comparison
        img_path, mask_path = matched_pairs[0]
        threshold_comparison(model, img_path, mask_path)
    
    elif choice == "4":
        # 10 örnek
        batch_skin_test(model, matched_pairs, num_samples=10)
    
    elif choice == "5":
        # Model bilgileri
        print("\n📋 Model Bilgileri:")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Model parametreleri: {model.count_params():,}")
        model.summary()
    
    else:
        print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    main()
