# MNIST Handwritten Digit Classification with Convolutional Neural Networks

This project demonstrates how to build a simple Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. The project is implemented using Keras and TensorFlow libraries.

## Table of Contents

1. [Installation](#installation)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Model Building](#model-building)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

First, ensure you have Python installed. Then, install the required libraries using pip:

```bash
pip install keras tensorflow matplotlib numpy
```

## Data Loading and Preprocessing

1. **Import Libraries**

    ```python
    from keras.datasets import mnist
    import tensorflow as tf
    import numpy as np
    from matplotlib import pyplot as plt
    ```

2. **Load the MNIST Dataset**

    ```python
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(y_train.shape, y_test.shape)
    ```

3. **Visualize the Data**

    ```python
    plt.imshow(X_train[0])
    ```

4. **Reshape Data**

    ```python
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    ```

5. **One-Hot Encode the Labels**

    ```python
    from keras.utils import to_categorical
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)
    print("y_train[0]:", y_train[0])
    print("y_train_onehot[0]:", y_train_onehot[0])
    ```

## Model Building

1. **Import Required Layers**

    ```python
    from keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    ```

2. **Define the Model**

    ```python
    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ])
    ```

3. **Compile the Model**

    ```python
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ```

## Model Training

1. **Train the Model**

    ```python
    model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=3)
    ```

## Model Evaluation

1. **Make Predictions**

    ```python
    pred_probs = model.predict(X_test)
    pred_probs[:4]
    ```

2. **Convert Predictions to Class Labels**

    ```python
    pred_classes = np.argmax(pred_probs, axis=1)
    pred_classes[:4]
    ```

3. **Compare with Actual Labels**

    ```python
    y_test[:4]
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This project provides a foundational approach to understanding and implementing a CNN using the MNIST dataset. You can expand upon this by experimenting with different architectures, hyperparameters, and optimization techniques.
