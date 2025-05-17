import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define paths
train_dir = 'data/train_test_datasets/train'
val_dir = 'data/train_test_datasets/validation'

# image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# data loaders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print details of the data generators
print(f"Training data: {train_generator.samples} images, {train_generator.num_classes} classes")
print(f"Validation data: {val_generator.samples} images, {val_generator.num_classes} classes")

# model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# print the model summary
print("\nModel Summary:")
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train model
print("\nTraining the model...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Print training results
print("\nTraining results:")
for epoch in range(10):
    print(f"Epoch {epoch+1}/{10}")
    print(f" - Loss: {history.history['loss'][epoch]}")
    print(f" - Accuracy: {history.history['accuracy'][epoch]}")
    print(f" - Val Loss: {history.history['val_loss'][epoch]}")
    print(f" - Val Accuracy: {history.history['val_accuracy'][epoch]}")

# Save model
model.save('models/moroccan_currency_classifier.keras')
print("\nModel saved as 'moroccan_currency_classifier.keras'")
