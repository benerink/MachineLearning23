import os
from tkinter import filedialog

import pandas as pd


def get_relative_path(directory, filename):
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join(current_directory, "..", directory, filename)
    return os.path.abspath(relative_path)


def get_excel_file_path(filename):
    resources_directory = "resources"
    file_path = get_relative_path(resources_directory, filename)

    try:
        if os.path.isfile(file_path):
            return file_path
        else:
            print(f"Error: File '{filename}' not found in the resources directory.")
            return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse Excel file '{filename}'. Make sure it's a valid Excel file.")
        return None


def upload_file():
    # Function to upload a file using tkinter file dialog
    file_path = filedialog.askopenfilename()

    # Extract filename and filesize
    filename = file_path.split('/')[-1]
    filesize = file_path.split('/')[-2]

    # Read the file based on its type (CSV or Excel)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df
