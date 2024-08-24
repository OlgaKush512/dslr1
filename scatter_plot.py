import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys

def wrap_text(text, width=15):
    return '\n'.join(textwrap.wrap(text, width=width))

def create_scatter_matrix(filename):
    # Reading data from CSV file with error handling
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

    # Selecting columns with subjects (after removing unnecessary columns)
    features = data.columns[6:]  # Columns with subjects starting from 'Arithmancy' to 'Flying'

    # Creating scatter plots for all pairs of subjects
    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20), sharex='col', sharey='row')

    # Creating scatter plots
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                # Creating an empty plot on the diagonal
                axes[i, j].plot([], [])  # Empty plot
            else:
                # Creating scatter plot manually
                axes[i, j].scatter(data[features[j]], data[features[i]], s=1)
            xticks = np.round(np.linspace(data[features[j]].min(), data[features[j]].max(), num=5), 3)  # Numeric values for X axis
            yticks = np.round(np.linspace(data[features[i]].min(), data[features[i]].max(), num=5), 3)  # Numeric values for Y axis
            axes[i, j].set_xticks(xticks)  # Numeric values for X axis
            axes[i, j].set_yticks(yticks)
            # Setting font for numeric values
            axes[i, j].set_xticklabels(xticks, fontsize=5, rotation=90)
            axes[i, j].set_yticklabels(yticks, fontsize=5)

    # Adding axis labels
    # Column titles (at the top)
    for j in range(num_features):
        axes[0, j].set_title(wrap_text(features[j]), fontsize=8, pad=10)

    # Row titles (on the left)
    for i in range(num_features):
        axes[i, 0].set_ylabel(wrap_text(features[i]), fontsize=8, labelpad=30, rotation=0)

    # Adjusting spacing between plots and displaying the plots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.show()

# Example usage: call the function with the name of your file
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <filename>")
    else:
        filename = sys.argv[1]
        create_scatter_matrix(filename)
