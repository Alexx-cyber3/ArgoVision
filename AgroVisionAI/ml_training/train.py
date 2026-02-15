import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import argparse

# Configuration for Maximum Accuracy
IMG_SIZE = (380, 380) # Optimal for EfficientNetB4
BATCH_SIZE = 16 # Reduced batch size for higher precision gradients
EPOCHS = 50 # More epochs for deep convergence
LEARNING_RATE = 0.0001

def create_expert_model(num_classes):
    print("Building Expert-Grade Architecture: EfficientNetB4 + Deep Head...")
    base_model = tf.keras.applications.EfficientNetB4(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Fine-tuning: Unfreeze the top layers of the base model
    base_model.trainable = True
    # We unfreeze the top 100 layers for specialized plant feature adaptation
    for layer in base_model.layers[:-100]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(model_name, data_dir, save_dir):
    print(f"--- AgroVision AI Expert Training Engine ---")
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found. Run download_data.py first.")
        return

    # Expert Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.1 # Using 10% for validation
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = train_generator.num_classes
    model = create_expert_model(num_classes)
    
    # Use Label Smoothing to improve generalization and accuracy
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # High-Precision Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, f"{model_name}.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=1e-7, verbose=1)

    print("Starting Deep Learning Phase...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print(f"
Training Complete. Best Model Saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--save", type=str, default="../web_ui/models")
    parser.add_argument("--name", type=str, default="expert_plant_model")
    
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)
    
    train_model(args.name, args.data, args.save)
