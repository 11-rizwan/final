import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Compute class weights
classes = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)
class_weight_dict = dict(enumerate(class_weights))

# Load base model
base_model = EfficientNetB5(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
base_model.trainable = False

# Build top layers
x = base_model.output
x = GlobalAveragePooling2D(name="gap_layer")(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid', name="output_layer")(x)

model = Model(inputs=base_model.input, outputs=out)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_pneumonia_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the top layers
print("Training top layers...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, earlystop]
)

# Optional: Fine-tune entire model
print("Fine-tuning base model...")
base_model.trainable = True

# Re-compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_finetune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, earlystop]
)

# Save final model
model.save("pneumonia_efficientnetb5_model.h5")
