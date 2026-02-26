import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Veri Hazırlama
dataset = "Drug Vision/Data combined"
image_dir = Path(dataset)
filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths_series = pd.Series(filepaths, name="filepath").astype("str")
labels_series = pd.Series(labels, name="label").astype("str")
image_df = pd.concat([filepaths_series, labels_series], axis=1)

train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42, shuffle=True)

# Jeneratörler
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)
test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=64,
    class_mode="categorical",
    subset="training"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=64,
    class_mode="categorical",
    subset="validation"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# Model
pretrained_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
pretrained_model.trainable = False

inputs = pretrained_model.input
x = pretrained_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])


checkpoint_path = "model_checkpoint.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=10, 
    callbacks=[checkpoint_callback, early_stopping]
)


plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend()
plt.show()


results = model.evaluate(test_images, verbose=0)
print(f"Test Kaybı: {results[0]:.4f}")
print(f"Test Doğruluğu: %{results[1]*100:.2f}")

pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in pred]

y_test = list(test_df.label)
print(classification_report(y_test, predictions))