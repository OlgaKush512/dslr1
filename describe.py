import csv
import math
import sys

def calculate_mean(values):
    return sum(values) / len(values)

def calculate_std(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def calculate_percentile(values, percentile):
    index = int(len(values) * percentile / 100)
    return sorted(values)[index]

def describe(file_path):
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            
            data = {header: [] for header in headers[6:]}  
            for row in reader:
                for i, value in enumerate(row[6:], start=6):
                    if value:  
                        data[headers[i]].append(float(value))
            
            metrics = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            results = {metric: [] for metric in metrics}
            
            for feature, values in data.items():
                count = len(values)
                mean = calculate_mean(values)
                std = calculate_std(values, mean)
                min_val = min(values)
                max_val = max(values)
                percentile_25 = calculate_percentile(values, 25)
                percentile_50 = calculate_percentile(values, 50)
                percentile_75 = calculate_percentile(values, 75)
                
                results['Count'].append(count)
                results['Mean'].append(mean)
                results['Std'].append(std)
                results['Min'].append(min_val)
                results['25%'].append(percentile_25)
                results['50%'].append(percentile_50)
                results['75%'].append(percentile_75)
                results['Max'].append(max_val)
            
            max_feature_length = max(len(feature) for feature in data.keys())
            column_width = min(max_feature_length + 4, 15)
            metric_width = max(len(metric) for metric in metrics) + 4
            
            
            def truncate_name(name, length):
                return name[:length].ljust(length)
            
            
            print(f"{'Metric'.ljust(metric_width)}", end="")
            for feature in data.keys():
                print(f"{truncate_name(feature, column_width)}", end="")
            print()
            
            
            for metric, values in results.items():
                print(f"{metric.ljust(metric_width)}", end="")
                for value in values:
                    print(f"{value:<{column_width}.6f}", end="") 
                print()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <path_to_csv_file>")
    else:
        file_path = sys.argv[1]
        describe(file_path)