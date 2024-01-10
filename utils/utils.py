import os
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

