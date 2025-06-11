import tensorflow as tf
import numpy as np
import os

os.makedirs("data", exist_ok=True)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)
print("✅ MNIST dataset downloaded and saved to /data")