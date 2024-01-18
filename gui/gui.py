import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import utils.utils
from models.find_top_features import find_top_features
from models.lineare_regression import train_and_predict
from models.random_forest import train_and_predict_random_forest


class FeatureSelectionWindow(ctk.CTkToplevel):
    def __init__(self, parent, top_features, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Feature Selection")
        self.lift()
        self.focus_set()
        self.top_features = top_features
        self.selected_feature = ctk.StringVar()

        label = ctk.CTkLabel(self, text="Select one of the top features:")
        label.pack(pady=10)

        # Create radio buttons for each top feature
        for feature in top_features:
            ctk.CTkRadioButton(self, text=feature, variable=self.selected_feature, value=feature).pack()

        ok_button = ctk.CTkButton(self, text="Okay", command=self.on_okay)
        ok_button.pack(pady=10)

    def on_okay(self):
        selected_feature = self.selected_feature.get()
        if selected_feature:
            print(f"Selected Feature: {selected_feature}")
            # Perform any further actions with the selected feature if needed
            self.destroy()
        else:
            print("Please select a feature.")

    def get_selected_feature(self):
        return self.selected_feature.get()


class TopFeaturesWindow(ctk.CTkToplevel):
    def __init__(self, parent, df, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.title("Top Features")
        self.lift()
        self.focus_set()
        self.df = df
        self.top_features = None

        input_label = ctk.CTkLabel(self, text="Enter the number of top features:")
        input_label.pack(pady=10)

        self.num_top_features_entry = ctk.CTkEntry(self)
        self.num_top_features_entry.pack(pady=10)

        ok_button = ctk.CTkButton(self, text="Okay", command=self.on_okay)
        ok_button.pack(pady=10)

    def on_okay(self):
        try:
            num_top_features = int(self.num_top_features_entry.get())
            self.top_features = find_top_features(self.df, num_top_features)
            print(f"Top {num_top_features} Features:", self.top_features)

        except ValueError:
            print("Please enter a valid integer for the number of top features.")
        self.destroy()
        # Open the FeatureSelectionWindow with the top features

    def get_top_features(self):
        return self.top_features


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
                                                   command=lambda: self.predict_model(model_type='Linear Regression'))
        self.sidebar_button_linear.grid(row=2, column=0, padx=20, pady=10, columnspan=2, sticky='nsew')

        self.sidebar_button_random_forest = ctk.CTkButton(self.sidebar_frame, corner_radius=8, text='Random Forest',
                                                          command=lambda: self.predict_model(
                                                              model_type='Random Forest'))
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

        # Entry for user to input the number of top features
        self.num_top_features_entry = ctk.CTkEntry(self.main_frame, corner_radius=8)
        self.num_top_features_entry.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.num_top_features_entry_label = ctk.CTkLabel(self.main_frame, text='Number of Top Features:')
        self.num_top_features_entry_label.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Label for displaying the size of the hovered point
        self.hover_label_var = ctk.StringVar()
        self.hover_label_var.set('Größe des  Punktes:')
        self.hover_label = ctk.CTkLabel(self.main_frame, textvariable=self.hover_label_var)
        self.hover_label.grid(row=0, column=0, padx=10, pady=15, sticky='w')

    def predict_model(self, model_type):
        df = utils.utils.upload_file()
        top_features = self.find_top_features_callback(df)
        selected_features = self.find_selected_features_callback(top_features)
        if model_type == 'Linear Regression':
            X_test, y_test, y_pred, predicted_price = train_and_predict(df, top_features)
            self.plot_data(X_test, y_test, y_pred, predicted_price, 'Linear Regression', top_features,
                           selected_features)
        elif model_type == 'Random Forest':
            X_test_rf, y_test_rf, y_pred_rf, predicted_price_rf = train_and_predict_random_forest(df, top_features)
            self.plot_data(X_test_rf, y_test_rf, y_pred_rf, predicted_price_rf, 'Random Forest', top_features,
                           selected_features)

    def plot_data(self, X_test, y_test, y_pred, predicted_price, model_type, features, selected_feature):
        plt.clf()
        plotpoints = 100
        fig, ax = plt.subplots()

        # Ensure at least one feature is provided
        if not features:
            raise ValueError("At least one feature must be provided.")

        # Scatter plot for actual values
        scatter_actual = ax.scatter(X_test[selected_feature], y_test, s=100, c='black', label='Actual Prices')

        # Scatter plot for predicted values
        if model_type == 'Linear Regression':
            scatter_pred = ax.scatter(X_test[selected_feature], y_pred, s=100, c='blue',
                                      label='Linear Regression Predicted Prices')
        elif model_type == 'Random Forest':
            scatter_pred = ax.scatter(X_test[selected_feature], y_pred, s=100, c='green',
                                      label='Random Forest Predicted Prices')

        # Add a point for the predicted selling price for new data
        if model_type == 'Linear Regression':
            ax.scatter(2025, predicted_price.item(), s=150, c='red', marker='X',
                       label='Linear Regression Predicted Price for 2025')  # Adjust the marker size (s)
        elif model_type == 'Random Forest':
            ax.scatter(2025, predicted_price.item(), s=150, c='orange', marker='X',
                       label='Random Forest Predicted Price for 2025')  # Adjust the marker size (s)

        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Verkaufspreis (Selling Price)')
        ax.legend()

        # Dynamically adjust the x-axis limits based on the selected feature
        min_x = min(X_test[selected_feature])
        max_x = max(X_test[selected_feature])
        ax.set_xlim(min_x, max_x)

        # Manually set the y-axis limits based on your expected range
        ax.set_ylim(0, 600000)

        fig.set_size_inches(25, 20)

        # Embed the plot into the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.tabview.tab("Plot1"))
        canvas.draw()
        canvas.get_tk_widget().grid(sticky='nsew', row=1, column=0)
        self.update()

    def find_top_features_callback(self, df):

        top_features = None

        try:
            top_features_window = TopFeaturesWindow(self, df)
            top_features_window.wait_window()
            top_features = top_features_window.get_top_features()


        except ValueError:
            print("An error occurred.")

        return top_features

    def find_selected_features_callback(self, df):

        selected_features = None

        try:

            selected_features_window = FeatureSelectionWindow(self, df)
            selected_features_window.wait_window()
            selected_features = selected_features_window.get_selected_feature()

        except ValueError:
            print("An error occurred.")
        print(selected_features)
        return selected_features
