import tkinter as tk

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import utils.utils
from models import random_forest, lineare_regression


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # root
        self.title("Hackerthon.py")
        self.geometry(f"{1920}x{1080}")
        self.minsize(500, 300)
        # root Layout
        self.columnconfigure(1, weight=8)
        self.columnconfigure(0, weight=0)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0, border_width=4)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

        # Sidebar Widgets
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Choose Model",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Sidebar buttons for different actions

        self.sidebar_button_linear = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Linear',
                                                       command=self.predict_linear)
        self.sidebar_button_linear.grid(row=2, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_random_forest = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Random Forest',
                                                         command=self.predict_random_forest)
        self.sidebar_button_random_forest.grid(row=3, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')


        # Mainframe creation
        self.main_frame = ctk.CTkFrame(self, )
        self.main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # create tabview
        self.tabview = ctk.CTkTabview(self.main_frame, fg_color='transparent')
        self.tabview.grid(row=1, columnspan=2, sticky="nsew")
        self.tabview.add("Plot1")
        self.tabview.tab("Plot1").grid(sticky='nsew')
        self.tabview.tab("Plot1").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Plot1").grid_rowconfigure(1, weight=1)


        # Top right

        # Label for displaying the size of the hovered point
        self.hover_label_var = ctk.StringVar()
        self.hover_label_var.set('Größe des  Punktes:')
        self.hover_label = ctk.CTkLabel(self.main_frame, textvariable=self.hover_label_var)
        self.hover_label.grid(row=0, column=0, padx=10, pady=15, sticky='w')

        # Dropdown menu for choosing dark mode
        darkmode_options = ['System', 'Dark', 'Light']
        self.selected_option = tk.StringVar()
        self.selected_option.set(darkmode_options[0])
        self.darkmode_dropdown_menu = ctk.CTkOptionMenu(self.main_frame, values=darkmode_options,
                                                        variable=self.selected_option, command=change_darkmode)
        self.darkmode_dropdown_menu.grid(row=0, column=1, sticky="ne", padx=10, pady=15)

    def update_window(self, X_test, y_test, y_pred, predicted_price):
        plotpoints = 100
        fig, ax = plt.subplots()

        # Scatter plot for actual values
        scatter_actual = ax.scatter(X_test.iloc[:, 0], y_test, s=400, c='black', label='Actual Prices')

        # Scatter plot for predicted values
        scatter_pred = ax.scatter(X_test.iloc[:, 0], y_pred, s=400, c='blue', label='Predicted Prices')

        # Add a point for the predicted selling price for new data
        ax.scatter(2025, predicted_price.item(), s=400, c='red', marker='X', label='Predicted Price for 2025')

        ax.set_xlabel('Baujahr (Year of Construction)')
        ax.set_ylabel('Verkaufspreis (Selling Price)')
        ax.legend()



        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # Adjust the size of the figure
        fig.set_size_inches(8, 4)  # Adjust width and height according to your preferences

        # Embed the plot into the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("Plot1"))
        canvas.draw()
        canvas.get_tk_widget().grid(sticky='nsew', row=1, column=0)
        self.update()

    def predict_linear(self):
        pd = utils.utils.upload_file()
        X_test, y_test, y_pred, predicted_price = lineare_regression.train_and_predict(pd)
        # Pass the values to the update_window function
        self.update_window(X_test, y_test, y_pred, predicted_price)

    def predict_random_forest(self):
        pd = utils.utils.upload_file()
        random_forest.train_random_forest()
        self.update_window()

# Function to change the appearance mode (dark mode, light mode, etc.)
def change_darkmode(choice):
    ctk.set_appearance_mode(choice)

