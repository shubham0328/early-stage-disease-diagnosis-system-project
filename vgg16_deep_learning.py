import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Define paths (replace test_image_path with your actual image file)
train_dir = 'dataset/train'  # Directory containing training images in subfolders by class
validation_dir = 'dataset/validation'  # Directory containing validation images in subfolders by class
test_dir = 'dataset/test'  # Directory containing test images in subfolders by class
test_image_path = 'dataset/test/class1/test_image.jpg'  # Replace 'test_image.jpg' with an actual image file name

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

# Preprocessing for validation and test (no augmentation)
validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

# Create generators for training, validation, and test data
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No shuffle for test set
)

# Get the number of classes and class labels from train generator
num_classes = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define steps per epoch, validation steps, and test steps to include all samples
steps_per_epoch = int(np.ceil(train_generator.samples / batch_size))
validation_steps = int(np.ceil(validation_generator.samples / batch_size))
test_steps = int(np.ceil(test_generator.samples / batch_size))

# Set up checkpoint to save the best model based on validation loss
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model for 30 epochs
print("Training the model for 30 epochs...")
history_30 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint]
)

# Extend training to 100 epochs
print("Continuing training to 100 epochs...")
history_100 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    initial_epoch=30,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint]
)

# Print final validation accuracy after 100 epochs
final_val_accuracy = history_100.history['val_accuracy'][-1]
print(f"Validation Accuracy after 100 epochs: {final_val_accuracy * 100:.2f}%")

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_30.history['accuracy'] + history_100.history['accuracy'], label='Training Accuracy')
plt.plot(history_30.history['val_accuracy'] + history_100.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_30.history['loss'] + history_100.history['loss'], label='Training Loss')
plt.plot(history_30.history['val_loss'] + history_100.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_plots.png')  # Save plot instead of showing it

# Load the best saved model
model = load_model('best_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Test the model with a single image
img = load_img(test_image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Predict the class
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_labels[predicted_class_index]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class name: {predicted_class_name}")