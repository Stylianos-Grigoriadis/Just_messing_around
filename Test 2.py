import tkinter as tk
from tkinter import filedialog
import pandas as pd


def open_file():
    # Create a file dialog to select a .csv file
    file_path = filedialog.askopenfilename(title="Select a CSV file",filetypes=[("CSV files", "*.xlsx"), ("All files", "*.*")])
    return file_path


# Create a basic Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Call the open_file function
file_path = open_file()
df = pd.read_excel(file_path)
print(df)