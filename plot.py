'''import matplotlib.pyplot as plt

def analyze_text_file(filename):
    element_counts = {}
    total_count = 0

    with open(filename, 'r') as file:
        for line in file:
            elements = line.strip().split()
            for element in elements:
                if element in element_counts:
                    element_counts[element] += 1
                else:
                    element_counts[element] = 1
                total_count += 1
    
    percentages = {element: (count / total_count) * 100 for element, count in element_counts.items()}
    
    labels = list(percentages.keys())
    values = list(percentages.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Elements')
    plt.ylabel('Percentage')
    plt.title('Percentage of Different Types of Elements')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    filename = input("Enter the name of the text file: ")
    analyze_text_file(filename)

if __name__ == "__main__":
    main()'''

import matplotlib.pyplot as plt
from collections import Counter

def analyze_list(input_list):
    # Count the elements in the list
    element_counts = Counter(input_list)
    total_count = len(input_list)
    
    # Calculate percentages
    percentages = {element: (count / total_count) * 100 for element, count in element_counts.items()}
    
    # Plot the percentages
    labels = list(percentages.keys())
    values = list(percentages.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Elements')
    plt.ylabel('Percentage')
    plt.title('Percentage of Different Types of Elements')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    # Example list (you can replace it with any list you want)
    input_list = ['apple', 'banana', 'apple', 'orange', 'apple', 'banana', 'orange', 'orange', 'grape']
    analyze_list(input_list)

if __name__ == "__main__":
    main()

