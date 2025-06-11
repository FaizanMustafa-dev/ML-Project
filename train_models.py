import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

os.makedirs("models", exist_ok=True)

# Train ML models on Wine Quality dataset
data = pd.read_csv("data/winequality-red.csv")
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models_ml = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models_ml.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.pkl")

# Train DL model on MNIST
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=3)
cnn_model.save("models/cnn_mnist.h5")
print("✅ CNN model trained and saved.")