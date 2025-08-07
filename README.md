# image_class_fashionMNIST
# Fashion MNIST Image Classification with Keras

This project is a simple image classification task using the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) and a basic neural network built with TensorFlow and Keras. The model is trained to classify grayscale 28x28 images into one of 10 clothing categories.

---

## Dataset

The dataset used is `fashion_mnist` from `tensorflow.keras.datasets`.

- 60,000 training images
- 10,000 testing images
- 10 categories of clothing items:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

---

## Model Architecture

```python
model = keras.Sequential([
    keras.Input(shape=(28, 28)),        # Input layer for 28x28 grayscale images
    keras.layers.Flatten(),             # Flatten the image into a 784-dim vector
    keras.layers.Dense(128, activation="relu"),  # Hidden dense layer with ReLU
    keras.layers.Dense(10, activation="softmax") # Output layer with 10 units (classes)
])
