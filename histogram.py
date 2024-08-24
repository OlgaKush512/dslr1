import csv
import sys
import matplotlib.pyplot as plt
import math

def load_data(file_path):
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            
            house_index = headers.index("Hogwarts House")
            course_indices = {header: i for i, header in enumerate(headers[6:])}
            
            data = {header: {house: [] for house in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]} for header in course_indices.keys()}
            
            for row in reader:
                house = row[house_index]
                for course, index in course_indices.items():
                    if row[index + 6]:  
                        data[course][house].append(float(row[index + 6]))
            
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for file '{file_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while opening the file: {e}")
        sys.exit(1)

def plot_histogram(data):
    num_courses = len(data)
    num_cols = 3  
    num_rows = math.ceil(num_courses / num_cols) 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))
    axes = axes.flatten()  
    
    for idx, (course, scores) in enumerate(data.items()):
        for house, house_scores in scores.items():
            axes[idx].hist(house_scores, bins=10, alpha=0.5, label=house)
        
        axes[idx].set_title(f'{course}')
        axes[idx].set_xlabel('Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True)
    
    
    if len(axes) > num_courses:
        for i in range(num_courses, len(axes)):
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.2,top=0.95) 
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <path_to_csv_file>")
        sys.exit(1)
    else:
        file_path = sys.argv[1]
        data = load_data(file_path)
        plot_histogram(data)
