import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load a pre-trained VGG19 model
model = VGG19(weights='imagenet', include_top=True)
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Use a smaller subset of images
subset_size = 50  # Reduce this if needed
x_train = x_train[:subset_size]
y_train = y_train[:subset_size]
x_test = x_test[:subset_size]
y_test = y_test[:subset_size]

# Resize CIFAR-10 images to 224x224
def resize_images(images, size):
    return np.array([tf.image.resize(image, size).numpy() for image in images])

x_train_resized = resize_images(x_train, (224, 224))
x_test_resized = resize_images(x_test, (224, 224))

# Extract filter responses in batches
def get_filter_responses(model, layer_names, data, batch_size=50):
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=[model.get_layer(name).output for name in layer_names])
    intermediate_output = []
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i:i + batch_size]
        batch_output = intermediate_layer_model.predict(batch_data)
        if not intermediate_output:
            intermediate_output = batch_output
        else:
            for j in range(len(intermediate_output)):
                intermediate_output[j] = np.concatenate((intermediate_output[j], batch_output[j]), axis=0)
    return intermediate_output

train_responses = get_filter_responses(model, layer_names, x_train_resized)
test_responses = get_filter_responses(model, layer_names, x_test_resized)

# Transformation function (Frobenius norm)
def frobenius_norm(response):
    transformed = []
    for resp in response:
        # Flatten each filter response to ensure uniform shape
        flattened_resp = [np.linalg.norm(filt, ord='fro') for filt in resp]
        transformed.append(flattened_resp)
    return np.array(transformed)

train_transformed = [frobenius_norm(layer_response) for layer_response in train_responses]
test_transformed = [frobenius_norm(layer_response) for layer_response in test_responses]

# Fit Structural Equations (using Linear Regression for simplicity)
def fit_structural_equations(transformed_responses):
    regressors = []
    for i in range(1, len(transformed_responses)):
        X = transformed_responses[i-1]
        y = transformed_responses[i]
        reg = LinearRegression().fit(X, y)
        regressors.append(reg)
    return regressors

regressors = fit_structural_equations(train_transformed)

# Sanity Check
def check_model_accuracy(regressors, test_transformed):
    predictions = test_transformed[0]
    for i in range(len(regressors)):
        predictions = regressors[i].predict(predictions)
    return mean_squared_error(test_transformed[-1], predictions)

mse = check_model_accuracy(regressors, test_transformed)
print(f'Mean Squared Error of the SCM: {mse}')

# Estimate Filter Importance
def estimate_filter_importance(regressors, test_transformed):
    baseline_accuracy = model.evaluate(x_test_resized, y_test, verbose=0)[1]
    importances = []
    for layer_index, reg in enumerate(regressors):
        for filter_index in range(reg.coef_.shape[1]):
            perturbed = np.copy(test_transformed[layer_index])
            perturbed[:, filter_index] = 0  # Set the filter response to zero
            predictions = perturbed
            for i in range(layer_index, len(regressors)):
                predictions = regressors[i].predict(predictions)
            perturbed_accuracy = model.evaluate(x_test_resized, y_test, verbose=0)[1]
            importance = baseline_accuracy - perturbed_accuracy
            importances.append((layer_names[layer_index], filter_index, importance))
        return sorted(importances, key=lambda x: x[2], reverse=True)

filter_importances = estimate_filter_importance(regressors, test_transformed)
for layer, filter_index, importance in filter_importances[:10]:  # Display top 10 important filters
    print(f'Layer: {layer}, Filter: {filter_index}, Importance: {importance}')