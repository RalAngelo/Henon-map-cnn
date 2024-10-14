# Hénon Map CNN Predictor

This project predicts the future states of the Hénon map using a Convolutional Neural Network (CNN) built with TensorFlow. The data is processed using Takens' embedding theorem, and the results can be visualized through a graphical user interface (GUI) built with tkinter.

## Features
- Generate synthetic data using the Hénon map equations.
- Apply Takens' embedding theorem to prepare the dataset for CNN.
- Build and train a CNN model to predict future states.
- Visualize the results with matplotlib and seaborn.
- GUI for user interaction to run the model and visualize predictions.

## Installation

1. Clone the repository:
git clone https://github.com/RalAngelo/Henon-map-cnn.git

2. Install the necessary dependencies:
pip install pandas numpy tensorflow matplotlib seaborn tk

3. Run the GUI:
python Henon.py

## How it works

- The Hénon map generates synthetic data which simulates chaotic dynamics.
- Takens' theorem is used to transform the data into a higher-dimensional space.
- A CNN is trained to predict future states based on the current state of the system.
- A tkinter GUI allows users to run the model with various parameters and view the results.