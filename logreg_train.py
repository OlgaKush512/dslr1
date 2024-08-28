import numpy as np
import pandas as pd
import argparse

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    return - (1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

def gradient_descent(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient

        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if np.isnan(theta).any() or np.isinf(theta).any():
            print(f"NaN or Inf detected at iteration {i}")
            break

    return theta, cost_history

def custom_mean(X):
    return sum(X) / len(X)

def custom_std(X, mean):
    variance = sum((x - mean) ** 2 for x in X)
    return (variance / len(X)) ** 0.5

def main(dataset_path):
    data = pd.read_csv(dataset_path)

    if data.isnull().values.any():
        print("Missing values found, filling them with the median.")
        data.fillna(data.median(numeric_only=True), inplace=True)

    X = data.iloc[:, 6:].values

    mean = [custom_mean(X[:, i]) for i in range(X.shape[1])]
    std = [custom_std(X[:, i], mean[i]) for i in range(X.shape[1])]

    X = (X - mean) / std

    y = data['Hogwarts House'].map({'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}).values

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    alpha = 0.001
    iterations = 10000
    num_classes = 4

    all_theta = np.zeros((num_classes, X.shape[1]))

    for i in range(num_classes):
        y_i = np.where(y == i, 1, 0)
        theta, _ = gradient_descent(X, y_i, alpha, iterations)
        all_theta[i] = theta

    np.savetxt('weights.csv', all_theta, delimiter=',')

    print("The model has been successfully trained, and the weights have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict Hogwarts houses.')
    parser.add_argument('dataset_path', type=str, help='Path to the training dataset file (CSV format).')

    args = parser.parse_args()

    main(args.dataset_path)
