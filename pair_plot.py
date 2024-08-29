import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

def load_csv(filename):
    dataset = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row in reader:
                new_row = []
                for value in row:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = np.nan
                    new_row.append(value)
                dataset.append(new_row)
        except csv.Error as e:
            print(f'file {filename}, line {reader.line_num}: {e}')
    return np.array(dataset, dtype=object)

def pair_plot_hist(ax, X):
    bins = [0, 327, 856, 1299, len(X)]
    colors = ['red', 'yellow', 'blue', 'green']
    for start, end, color in zip(bins[:-1], bins[1:], colors):
        h = X[start:end]
        h = h[~np.isnan(h)]
        ax.hist(h, alpha=0.5, color=color)

def pair_plot_scatter(ax, X, y):
    bins = [0, 327, 856, 1299, len(X)]
    colors = ['red', 'yellow', 'blue', 'green']
    for start, end, color in zip(bins[:-1], bins[1:], colors):
        ax.scatter(X[start:end], y[start:end], s=1, color=color, alpha=0.5)

def pair_plot(dataset, features, legend):
    font = {'family': 'DejaVu Sans', 'weight': 'light', 'size': 7}
    matplotlib.rc('font', **font)

    size = dataset.shape[1]
    fig, axes = plt.subplots(nrows=size, ncols=size, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    for row in range(size):
        for col in range(size):
            X = dataset[:, col]
            y = dataset[:, row]

            ax = axes[row, col]
            if col == row:
                pair_plot_hist(ax, X)
            else:
                pair_plot_scatter(ax, X, y)

            if row == size - 1:
                ax.set_xlabel(features[col].replace(' ', '\n'))
            else:
                ax.tick_params(labelbottom=False)

            if col == 0:
                ax.set_ylabel(features[row].replace(' ', '\n'))
            else:
                ax.tick_params(labelleft=False)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    plt.legend(legend, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <path_to_csv_file>")
        sys.exit(1)
    else:
        file_path = sys.argv[1]
        dataset = load_csv(file_path)
        data = dataset[1:, 6:]
        data = data[data[:, 1].argsort()]

        features = dataset[0, 6:]
        legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        pair_plot(np.array(data, dtype=float), features, legend)

if __name__ == '__main__':
    main()
