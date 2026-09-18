"""
AI-Based Pneumonia Detection Using Attention-Enhanced CNN
Training Pipeline and Model Architecture
"""

import os
import argparse
import numpy as np
import tensorflow as tf

try:
    import keras
    from keras import layers
    register_serializable = keras.saving.register_keras_serializable
except (ImportError, AttributeError):
    from tensorflow import keras
    from tensorflow.keras import layers
    register_serializable = tf.keras.utils.register_keras_serializable

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)

# ==============================================================================
# Attention Layers (CBAM: Channel and Spatial Attention)
# ==============================================================================

@register_serializable(package="PneumoniaAttention")
class ChannelAttention(layers.Layer):
    """
    Channel Attention Module:
    Computes global average and max pooling per channel, passes through a shared MLP,
    and applies a sigmoid gate to reweight feature channels.
    """
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.ratio, 4)
        self.fc1 = layers.Dense(reduced_channels, activation="relu", use_bias=True)
        self.fc2 = layers.Dense(channels, activation="sigmoid", use_bias=True)
        super().build(input_shape)

    def call(self, inputs):
        # Global Avg and Max across spatial dimensions (H, W)
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))

        attention_weights = tf.sigmoid(avg_out + max_out)
        return inputs * attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


@register_serializable(package="PneumoniaAttention")
class SpatialAttention(layers.Layer):
    """
    Spatial Attention Module:
    Applies channel-wise average and max pooling, concatenates them,
    and uses a 7x7 convolution with sigmoid activation to produce a 2D spatial attention mask.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="spatial_attention_conv"
        )

    def build(self, input_shape):
        self.conv.build((input_shape[0], input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def call(self, inputs):
        # Pool across channels (axis=-1)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_mask = self.conv(concat)
        return inputs * spatial_mask

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


# ==============================================================================
# Model Architecture
# ==============================================================================

def build_attention_enhanced_cnn(input_shape=(224, 224, 3)):
    """
    Builds the complete Attention-Enhanced CNN architecture for Pneumonia Detection.
    """
    inputs = layers.Input(shape=input_shape, name="input_xray")

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Attention Block: Channel Attention followed by Spatial Attention
    x = ChannelAttention(ratio=8, name="channel_attention")(x)
    x = SpatialAttention(kernel_size=7, name="spatial_attention")(x)

    # Convolutional Block 4 (Target conv layer for Grad-CAM tracking)
    x = layers.Conv2D(256, (3, 3), padding="same", name="target_conv_layer")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Pneumonia_Attention_CNN")
    return model


# ==============================================================================
# Data Loading & Augmentation
# ==============================================================================

def get_data_augmentation():
    """Returns data augmentation sequential layer for chest radiographs."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ], name="data_augmentation")


def count_files_in_dir(path):
    """Counts image files inside a folder recursively."""
    if not os.path.exists(path):
        return 0
    count = 0
    valid_exts = {".jpg", ".jpeg", ".png"}
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f.lower())[1] in valid_exts:
                count += 1
    return count


def load_datasets(dataset_dir="dataset", img_size=(224, 224), batch_size=32):
    """
    Loads train, validation, and test datasets from standard directory structure:
      dataset/train/NORMAL and PNEUMONIA
      dataset/validation/NORMAL and PNEUMONIA
      dataset/test/NORMAL and PNEUMONIA
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")

    train_count = count_files_in_dir(train_dir)
    val_count = count_files_in_dir(val_dir)
    test_count = count_files_in_dir(test_dir)

    print(f"[Dataset] Train images: {train_count}")
    print(f"[Dataset] Validation images: {val_count}")
    print(f"[Dataset] Test images: {test_count}")

    if train_count == 0:
        return None, None, None, 0, 0

    # Normalization rescaling layer
    normalization_layer = layers.Rescaling(1.0 / 255)
    augmentation = get_data_augmentation()

    # Load train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    # Class counts for weighting
    normal_train = count_files_in_dir(os.path.join(train_dir, "NORMAL"))
    pneumonia_train = count_files_in_dir(os.path.join(train_dir, "PNEUMONIA"))

    # Apply preprocessing & augmentation to train
    train_ds = train_ds.map(lambda x, y: (augmentation(normalization_layer(x), training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load validation dataset if available
    val_ds = None
    if val_count > 0:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="binary",
            class_names=["NORMAL", "PNEUMONIA"],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Load test dataset if available
    test_ds = None
    if test_count > 0:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="binary",
            class_names=["NORMAL", "PNEUMONIA"],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, normal_train, pneumonia_train


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train(epochs=20, batch_size=32, lr=1e-4, img_size=(224, 224), output_path="model/pneumonia_attention_cnn.keras", init_only=False):
    """
    Constructs, compiles, and optionally trains the Attention-Enhanced CNN.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "=" * 65)
    print("AI-Based Pneumonia Detection: Attention-Enhanced CNN")
    print("=" * 65)
    print(f"Target model output: {output_path}")
    print(f"Image resolution:    {img_size[0]}x{img_size[1]}")

    # Build architecture
    model = build_attention_enhanced_cnn(input_shape=(img_size[0], img_size[1], 3))

    # Compile with Adam and Binary Crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )

    model.summary()

    if init_only:
        print("\n[Mode] Initialization only requested.")
        model.save(output_path)
        print(f"Successfully created and saved initialized Attention-Enhanced CNN to {output_path}")
        return model

    # Check datasets
    train_ds, val_ds, test_ds, n_norm, n_pneu = load_datasets(
        dataset_dir="dataset",
        img_size=img_size,
        batch_size=batch_size
    )

    if train_ds is None:
        print("\n[Notice] No training images detected in dataset/train/NORMAL or dataset/train/PNEUMONIA.")
        print("Saving compiled model structure to ensure immediate web application availability...")
        model.save(output_path)
        print(f"Saved initialized model to {output_path}.")
        print("\nTo train on full data:")
        print("1. Download the Kaggle Chest X-Ray dataset (e.g. Kermany et al.)")
        print("2. Place radiographs into dataset/train/, dataset/validation/, dataset/test/")
        print("3. Run: python train.py --epochs 20")
        return model

    # Compute class weights for imbalance
    total_samples = n_norm + n_pneu
    weight_for_0 = (1 / n_norm) * (total_samples / 2.0) if n_norm > 0 else 1.0
    weight_for_1 = (1 / n_pneu) * (total_samples / 2.0) if n_pneu > 0 else 1.0
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"\n[Class Weights] Normal: {weight_for_0:.2f} | Pneumonia: {weight_for_1:.2f}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            monitor="val_loss" if val_ds is not None else "loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n[Training] Commencing model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights if n_norm > 0 and n_pneu > 0 else None,
        verbose=1
    )

    print(f"\n[Done] Training complete. Best model weights saved to {output_path}")

    # Test evaluation
    if test_ds is not None:
        print("\n[Evaluation] Evaluating on test dataset...")
        eval_results = model.evaluate(test_ds, verbose=1)
        print(f"Test Loss:     {eval_results[0]:.4f}")
        print(f"Test Accuracy: {eval_results[1]:.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention-Enhanced CNN for Pneumonia Detection")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--output", type=str, default="model/pneumonia_attention_cnn.keras", help="Model save path")
    parser.add_argument("--init-only", action="store_true", help="Build and save architecture without training")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=(args.img_size, args.img_size),
        output_path=args.output,
        init_only=args.init_only
    )
