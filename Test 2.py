import os

# Define the folder and file paths
folder_path = r'C:\Users\Stylianos\Desktop\name_change'  # Replace with your folder path
old_file_name = 'Isometric_05_T1.csv'  # Replace with the current CSV file name
new_file_name = 'Smoothly done Isometric_05_T1.csv'  # Replace with the new CSV file name

# Full paths
old_file_path = os.path.join(folder_path, old_file_name)
new_file_path = os.path.join(folder_path, new_file_name)
os.rename(old_file_path, new_file_path)

