import numpy as np
import pandas as pd
import argparse

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_one_vs_all(X, all_theta):
    probs = sigmoid(X @ all_theta.T)
    return np.argmax(probs, axis=1)

def custom_mean(X):
    return sum(X) / len(X)

def custom_std(X, mean):
    variance = sum((x - mean) ** 2 for x in X) / len(X)
    return variance ** 0.5

def main(weights_path, dataset_path):
    data = pd.read_csv(dataset_path)

    if data.isnull().values.any():
        print("Missing values found, filling them with the median.")
        data.fillna(data.median(numeric_only=True), inplace=True)

    X = data.iloc[:, 6:].values
    print(f"Example of the data after loading: {X[:5]}")

    indices = data['Index']

    mean = [custom_mean(X[:, i]) for i in range(X.shape[1])]
    std = [custom_std(X[:, i], mean[i]) for i in range(X.shape[1])]
    X = (X - mean) / std

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    all_theta = np.loadtxt(weights_path, delimiter=',')

    if all_theta.shape[1] != X.shape[1]:
        raise ValueError(f"The dimensions of the weights and features do not match: {all_theta.shape[1]} != {X.shape[1]}")

    predictions = predict_one_vs_all(X, all_theta)

    houses = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
    predicted_houses = [houses[p] for p in predictions]

    output = pd.DataFrame({'Index': indices, 'Hogwarts House': predicted_houses})
    output.to_csv('houses.csv', index=False)

    print("Predictions have been successfully saved to the file houses.csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Hogwarts houses from a dataset.')
    parser.add_argument('weights_path', type=str, help='Path to the weights file (CSV format).')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset file (CSV format).')

    args = parser.parse_args()

    main(args.weights_path, args.dataset_path)