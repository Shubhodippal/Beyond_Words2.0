import streamlit as st
import matplotlib.pyplot as plt

# Read elements from the .txt file
def read_elements_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        elements = [line.strip() for line in lines]
    return elements

# Calculate percentage of each element
def calculate_percentage(elements):
    total_elements = len(elements)
    element_counts = {}
    for element in elements:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    
    percentages = {}
    for element, count in element_counts.items():
        percentages[element] = (count / total_elements) * 100
    
    return percentages

# Plot pie chart
def plot_pie_chart(percentages):
    labels = percentages.keys()
    sizes = percentages.values()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title('Element Percentages')
    return fig

def main():
    st.title("Analyzed Result")
    analyze_result = st.button('Analyze')

    file_name = ("output.txt")
    if(analyze_result):
        if file_name is not None:
            elements = read_elements_from_file(file_name)
            st.write("Elements in the file:", elements)
            percentages = calculate_percentage(elements)
            st.write("Percentage of each element:", percentages)
            fig = plot_pie_chart(percentages)
            st.pyplot(fig)

if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
