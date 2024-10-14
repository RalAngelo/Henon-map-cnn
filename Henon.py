import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
import matplotlib.pyplot as plt

# Define the Henon map
def henon_map(x, y, a=1.4, b=0.3):
    x_new = 1 - a * x**2 + y
    y_new = b * x
    return x_new, y_new

# Generate Henon map data
def generate_henon_data(n_samples=1000):
    x, y = 0, 0
    data = []
    for _ in range(n_samples):
        x, y = henon_map(x, y)
        data.append([x, y])
    return np.array(data)

# Takens embedding function
def takens_embedding(data, delay, dimension):
    embedded_data = []
    for i in range(len(data) - delay * (dimension - 1)):
        embedded_data.append(np.array([data[i + j * delay] for j in range(dimension)]))
    return np.array(embedded_data)

# Build the CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(2)  # Output layer for predicting x and y
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the CNN model
def train_model(n_samples):
    # Generate data
    data = generate_henon_data(n_samples)
    
    # Takens embedding
    embedded_data = takens_embedding(data[:, 0], delay=1, dimension=3)
    
    # Prepare input/output for the model
    X = embedded_data[:-1]
    y = data[1:len(embedded_data)]
    
    # Reshape data for CNN input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build the model
    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
    
    return model, history, X_test, y_test

# Plot results
def plot_results(history, model, X_test, y_test):
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Predict and plot true vs predicted values
    predicted = model.predict(X_test)
    plt.figure(figsize=(10, 5))
    plt.plot(predicted[:100, 0], label='Predicted x')
    plt.plot(y_test[:100, 0], label='True x')
    plt.title('True vs Predicted - x coordinate')
    plt.legend()
    plt.show()

# GUI class
class HenonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Henon Map CNN Predictor")
        
        # Widgets
        self.label = tk.Label(root, text="Enter number of samples:")
        self.label.pack()
        self.root.geometry("400x100")

        self.entry = tk.Entry(root)
        self.entry.pack()

        self.button = tk.Button(root, text="Run Prediction", command=self.run_prediction)
        self.button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def run_prediction(self):
        try:
            n_samples = int(self.entry.get())
            if n_samples <= 0:
                raise ValueError("Samples must be positive.")
            
            # Train the model and get results
            model, history, X_test, y_test = train_model(n_samples)
            
            # Display success message
            self.result_label.config(text=f"Prediction for {n_samples} samples done!")

            # Plot the results
            plot_results(history, model, X_test, y_test)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

# Start the tkinter GUI
root = tk.Tk()
app = HenonApp(root)
root.mainloop()
