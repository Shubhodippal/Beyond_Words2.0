import tkinter as tk
from tkinter import messagebox
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FileAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Analyzer")

        """self.label = tk.Label(root, text="Enter the name of the text file:")
        self.label.pack()

        self.entry = tk.Entry(root)
        self.entry.pack()"""

        self.button = tk.Button(root, text="Analyze File", command=self.analyze_file)
        self.button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def analyze_file(self):
        file_name = "output.txt"
        try:
            with open(file_name, 'r') as file:
                content = file.read()
                elements = content.split()
                total_elements = len(elements)
                element_counts = Counter(elements)

                percentages = {element: (count / total_elements) * 100 for element, count in element_counts.items()}

                # Clear existing plot
                self.ax.clear()

                # Plot bar graph
                self.ax.bar(percentages.keys(), percentages.values())
                self.ax.set_ylabel('Percentage')
                self.ax.set_title('Percentage of Different Elements')

                # Draw the canvas
                self.canvas.draw()

        except FileNotFoundError:
            messagebox.showerror("Error", f"File '{file_name}' not found.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileAnalyzerApp(root)
    root.mainloop()
