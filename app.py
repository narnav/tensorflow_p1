import tensorflow as tf
from tensorflow.keras import layers, models


# Define the paths to your training and validation data
train_data_dir = 'train/'
validation_data_dir = 'valid'

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Use ImageDataGenerator to load and preprocess the images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,        # Rescale pixel values between 0 and 1
    shear_range=0.2,        # Apply random shear augmentation
    zoom_range=0.2,         # Apply random zoom augmentation
    horizontal_flip=True    # Flip images horizontally randomly
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# model
model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the 3D output to 1D
model.add(layers.Flatten())

# Add dense layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting
model.add(layers.Dense(number_of_classes, activation='softmax'))

# compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
epochs = 10  # You can adjust the number of epochs based on your dataset size and complexity

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# 
# You can evaluate the model on your test set if available
test_data_dir = 'path/to/test_data'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test accuracy:", test_accuracy)

