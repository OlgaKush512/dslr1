import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys

def wrap_text(text, width=15):
    return '\n'.join(textwrap.wrap(text, width=width))

def create_scatter_matrix(filename):
    try:
        data = pd.read_csv(filename)
        print("Data successfully loaded.")
    except FileNotFoundError:
        print("File not found. Please check the filename and try again.")
        return
    except pd.errors.EmptyDataError:
        print("File is empty.")
        return
    except pd.errors.ParserError:
        print("Error parsing the file. Please check the CSV file structure.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    features = data.columns[6:] 

    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20), sharex='col', sharey='row')

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axes[i, j].plot([], [])  
            else:
                axes[i, j].scatter(data[features[j]], data[features[i]], s=1)
            xticks = np.round(np.linspace(data[features[j]].min(), data[features[j]].max(), num=5), 3)  # Numeric values for X axis
            yticks = np.round(np.linspace(data[features[i]].min(), data[features[i]].max(), num=5), 3)  # Numeric values for Y axis
            axes[i, j].set_xticks(xticks)  
            axes[i, j].set_yticks(yticks)
            axes[i, j].set_xticklabels(xticks, fontsize=5, rotation=90)
            axes[i, j].set_yticklabels(yticks, fontsize=5)

    for j in range(num_features):
        axes[0, j].set_title(wrap_text(features[j]), fontsize=8, pad=10)

    for i in range(num_features):
        axes[i, 0].set_ylabel(wrap_text(features[i]), fontsize=8, labelpad=30, rotation=0)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <filename>")
    else:
        filename = sys.argv[1]
        create_scatter_matrix(filename)
