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

def mini_batch_gradient_descent(X, y, alpha, iterations, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            end = j + batch_size
            X_batch = X_shuffled[j:end]
            y_batch = y_shuffled[j:end]

            h = sigmoid(X_batch @ theta)
            gradient = (X_batch.T @ (h - y_batch)) / batch_size
            theta -= alpha * gradient

        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if np.isnan(theta).any() or np.isinf(theta).any():
            print(f"NaN or Inf detected at iteration {i}")
            return theta, cost_history

    return theta, cost_history

def custom_mean(X):
    return sum(X) / len(X)

def custom_std(X, mean):
    variance = sum((x - mean) ** 2 for x in X)
    return (variance / len(X)) ** 0.5

def main(dataset_path, exclude_columns):
    data = pd.read_csv(dataset_path)

    if data.isnull().values.any():
        data.fillna(data.median(numeric_only=True), inplace=True)

    X = data.iloc[:, 6:].values

    if exclude_columns:
        exclude_indices = [int(i) - 6 for i in exclude_columns]
        X = np.delete(X, exclude_indices, axis=1)


    mean = [custom_mean(X[:, i]) for i in range(X.shape[1])]
    std = [custom_std(X[:, i], mean[i]) for i in range(X.shape[1])]

    X = (X - mean) / std

    y = data['Hogwarts House'].map({'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}).values

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    alpha = 0.001
    iterations = 1000
    batch_size = 32
    num_classes = 4

    all_theta = np.zeros((num_classes, X.shape[1]))

    for i in range(num_classes):
        y_i = np.where(y == i, 1, 0)
        theta, _ = mini_batch_gradient_descent(X, y_i, alpha, iterations, batch_size)
        all_theta[i] = theta

    np.savetxt('weights.csv', all_theta, delimiter=',')

    print("The model has been successfully trained, and the weights have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict Hogwarts houses.')
    parser.add_argument('dataset_path', type=str, help='Path to the training dataset file (CSV format).')
    parser.add_argument('--exclude', type=int, nargs='*', help='Column indices to exclude from the features.')

    args = parser.parse_args()

    main(args.dataset_path,args.exclude)
