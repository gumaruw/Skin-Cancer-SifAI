import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os

# Collect all image file paths
train_images = glob.glob('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/*.jpg') + glob.glob('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/*.jpg')
train_masks = glob.glob('../input/ham10000-lesion-segmentations/HAM10000_segmentations_lesion_tschandl/*.png')

print(f"Total images: {len(train_images)}")
print(f"Total masks: {len(train_masks)}")

# Sort images and masks based on the filename without extensions
def sort_by_filename(images, masks):
    # Extract filenames without extensions
    image_filenames = [os.path.splitext(os.path.basename(img))[0] for img in images]
    mask_filenames = [os.path.splitext(os.path.basename(mask))[0] for mask in masks]

    # Sort both the images and masks based on the filenames
    sorted_image_paths = [img for _, img in sorted(zip(image_filenames, images))]
    sorted_mask_paths = [mask for _, mask in sorted(zip(mask_filenames, masks))]
    
    return sorted_image_paths, sorted_mask_paths

# Sort images and masks
train_images, train_masks = sort_by_filename(train_images, train_masks)

# Function to load and preprocess an image
def load_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode as RGB
    image = tf.image.resize(image, (256, 256))  # Resize if needed
    image = image / 255.0  # Normalize to [0,1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # Decode as grayscale
    mask = tf.image.resize(mask, (256, 256))  # Resize if needed
    mask = mask / 255.0  # Normalize

    return image, mask

# Create a TensorFlow dataset (lazy loading)
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)  # Load in parallel
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # Process in batches, optimize performance


# Split the dataset into train and test sets
train_size = int(0.8 * len(train_images))  # 80% for training
test_size = len(train_images) - train_size  # 20% for testing

# Split using `take` and `skip`
train_dataset = dataset.take(train_size // 32)  # Take first 80% for training
test_dataset = dataset.skip(train_size // 32)  # Skip the first 80% for testing

# Show input shapes for the first batch
for images, masks in dataset.take(1):
    print("Shape of images in the first batch:", images.shape)  # (batch_size, height, width, channels)
    print("Shape of masks in the first batch:", masks.shape)  # (batch_size, height, width)

# Function to plot images and their corresponding masks
def plot_images_and_masks(dataset, num_samples=3):
    plt.figure(figsize=(10, 10))
    
    for i, (images, masks) in enumerate(dataset.take(1)):
        for j in range(num_samples):
            ax1 = plt.subplot(num_samples, 2, 2 * j + 1)
            ax1.set_title("Input Image")
            plt.imshow(images[j])  # Display the image
            ax1.axis("off")

            ax2 = plt.subplot(num_samples, 2, 2 * j + 2)
            ax2.set_title("Ground Truth Mask")
            plt.imshow(masks[j], cmap='gray')  # Display the mask in grayscale
            ax2.axis("off")
        
        plt.show()
        break  # Display only the first batch

# Call the function to display some images and masks
plot_images_and_masks(train_dataset, num_samples=3)

# Define image size and channels
IMG_HEIGHT = 256  # Adjust this according to your data
IMG_WIDTH = 256   # Adjust this according to your data
IMG_CHANNELS = 3  # 1 for grayscale, 3 for RGB

# Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction path (same as before)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path (same as before)
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Model compilation
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.summary()
import tensorflow as tf

# Pixel Accuracy (PA)
def pixel_accuracy(y_true, y_pred):
    # Compare the predicted mask with the ground truth mask
    correct_pixels = tf.equal(y_true, tf.round(y_pred))
    accuracy = tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    return accuracy

# Intersection over Union (IoU)
def iou(y_true, y_pred):
    # Flatten the images to 1D arrays
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.round(y_pred), [-1])  # Round predictions to 0 or 1
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return intersection / union

# Dice Coefficient
def dice_coefficient(y_true, y_pred):
    # Flatten the images to 1D arrays
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.round(y_pred), [-1])  # Round predictions to 0 or 1
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 2. * intersection / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

# Compile the model with the custom metrics
model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=[pixel_accuracy, iou, dice_coefficient]
)

# Calculate number of steps per epoch and validation steps
steps_per_epoch = len(train_images) // 32
validation_steps = len(train_images) // 32

# Ensure the dataset repeats and does not run out of data
train_dataset = train_dataset.repeat()  # Repeat the dataset for multiple epochs
test_dataset = test_dataset.repeat()  # Repeat the validation dataset as well

from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback to monitor Pixel Accuracy (PA)
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    mode='min',  # 'max' to maximize PA
    patience=5,  # Stop after 5 epochs without improvement
    restore_best_weights=True  # Restore best weights after stopping
)

# Train the model
history = model.fit(
    train_dataset,  # Training dataset
    epochs=50,  # Set higher epochs to allow for more training
    validation_data=test_dataset,  # Validation dataset
    steps_per_epoch=steps_per_epoch,  # Number of steps per epoch
    validation_steps=validation_steps,  # Number of validation steps
    callbacks=[early_stopping]  # Add early stopping callback
)

# Function to plot images, predicted masks, and ground truth masks
def plot_predictions(model, dataset, num_samples=3):
    plt.figure(figsize=(10, 10))
    
    for images, masks in dataset.take(1):
        # Get predictions from the model
        predictions = model.predict(images)

        # Plot the first `num_samples` samples
        for i in range(num_samples):
            # Plot the input image
            ax1 = plt.subplot(num_samples, 3, 3 * i + 1)
            ax1.set_title("Input Image")
            plt.imshow(images[i])  # Display the image
            ax1.axis("off")

            # Plot the predicted mask
            ax2 = plt.subplot(num_samples, 3, 3 * i + 2)
            ax2.set_title("Predicted Mask")
            plt.imshow(tf.squeeze(predictions[i]), cmap='gray')  # Use tf.squeeze
            ax2.axis("off")

            # Plot the ground truth mask
            ax3 = plt.subplot(num_samples, 3, 3 * i + 3)
            ax3.set_title("Ground Truth Mask")
            plt.imshow(tf.squeeze(masks[i]), cmap='gray')  # Use tf.squeeze
            ax3.axis("off")

        plt.show()

# Call the function to display some predictions
plot_predictions(model, test_dataset, num_samples=3)

# Function to apply thresholding (rounding values to 0 or 1)
def threshold_image(image, threshold=0.5):
    # Convert the image to binary: values > threshold become 1, others become 0
    return np.where(image > threshold, 1.0, 0.0)

# Function to plot input images, thresholded predicted masks, and ground truth masks
def plot_predictions(model, dataset, num_samples=3):
    plt.figure(figsize=(12, 12), dpi=200)  # Adjusted figure size and dpi for clarity
    
    for images, masks in dataset.take(1):
        # Get predictions from the model
        predictions = model.predict(images)

        # Plot the first `num_samples` samples
        for i in range(num_samples):
            # Plot the input image
            ax1 = plt.subplot(num_samples, 3, 3 * i + 1)
            ax1.set_title("Input Image")
            plt.imshow(tf.squeeze(images[i]).numpy())  # Display the input image
            ax1.axis("off")

            # Plot the thresholded predicted mask
            ax2 = plt.subplot(num_samples, 3, 3 * i + 2)
            ax2.set_title("Thresholded Predicted Mask")
            thresholded_pred = threshold_image(tf.squeeze(predictions[i]).numpy(), threshold=0.5)
            plt.imshow(thresholded_pred, cmap='gray')  # Use 'gray' colormap for binary masks
            ax2.axis("off")

            # Plot the ground truth mask
            ax3 = plt.subplot(num_samples, 3, 3 * i + 3)
            ax3.set_title("Ground Truth Mask")
            plt.imshow(tf.squeeze(masks[i]).numpy(), cmap='gray')  # Ground truth in grayscale
            ax3.axis("off")

        plt.show()

# Call the function to display input images, thresholded predicted masks, and ground truth masks
plot_predictions(model, test_dataset, num_samples=3)

model.save('/kaggle/working/unet_lesion_model.h5')

# Test set evaluation
test_loss, test_pa, test_iou, test_dice = model.evaluate(test_dataset, steps=validation_steps)
print(f"Test Accuracy: {test_pa:.4f}")
print(f"Test IoU: {test_iou:.4f}")
print(f"Test Dice: {test_dice:.4f}")

# Evaluating the model
# Mevcut verisetinden rastgele görüntüler seç ve tahmin yap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def test_model_on_random_images(model, num_samples=6):
    """
    Modeli rastgele seçilen görüntülerde test et
    """
    # Tüm görüntü yollarından rastgele seç
    random_indices = np.random.choice(len(train_images), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(random_indices):
        # Rastgele bir görüntü ve mask yükle
        image_path = train_images[idx]
        mask_path = train_masks[idx]
        
        # Görüntüyü yükle ve önişle
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (256, 256))
        image = image / 255.0
        
        # Mask yükle
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (256, 256))
        mask = mask / 255.0
        
        # Tahmin yap
        prediction = model.predict(tf.expand_dims(image, axis=0))[0]
        
        # Binary mask yap
        binary_pred = (prediction > 0.5).astype(np.float32)
        
        # Görselleştir
        ax1 = plt.subplot(num_samples, 3, 3 * i + 1)
        ax1.set_title("Orijinal Görüntü")
        plt.imshow(image)
        ax1.axis("off")
        
        ax2 = plt.subplot(num_samples, 3, 3 * i + 2)
        ax2.set_title("Tahmin Edilen Mask")
        plt.imshow(tf.squeeze(binary_pred), cmap='gray')
        ax2.axis("off")
        
        ax3 = plt.subplot(num_samples, 3, 3 * i + 3)
        ax3.set_title("Gerçek Mask")
        plt.imshow(tf.squeeze(mask), cmap='gray')
        ax3.axis("off")
    
    plt.tight_layout()
    plt.show()

# Test et
print("Model test ediliyor...")
test_model_on_random_images(model, num_samples=6)

# Hızlı performans testi
def quick_performance_test(model, num_test=50):
    """
    Rastgele seçilen görüntülerde hızlı performans testi
    """
    random_indices = np.random.choice(len(train_images), num_test, replace=False)
    
    ious = []
    dices = []
    
    for idx in random_indices:
        # Görüntü ve mask yükle
        image_path = train_images[idx]
        mask_path = train_masks[idx]
        
        image, mask = load_image(image_path, mask_path)
        
        # Tahmin yap
        prediction = model.predict(tf.expand_dims(image, axis=0))[0]
        binary_pred = (prediction > 0.5).astype(np.float32)
        
        # IoU hesapla
        intersection = tf.reduce_sum(mask * binary_pred)
        union = tf.reduce_sum(mask) + tf.reduce_sum(binary_pred) - intersection
        iou_score = intersection / (union + 1e-7)
        
        # Dice hesapla
        dice_score = 2 * intersection / (tf.reduce_sum(mask) + tf.reduce_sum(binary_pred) + 1e-7)
        
        ious.append(float(iou_score))
        dices.append(float(dice_score))
    
    print(f"\n=== {num_test} Görüntüde Hızlı Test Sonuçları ===")
    print(f"Ortalama IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Ortalama Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Min IoU: {np.min(ious):.4f}")
    print(f"Max IoU: {np.max(ious):.4f}")

# Hızlı performans testi yap
quick_performance_test(model, num_test=50)
