# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:38:59 2023

@author: tyrel
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:38:59 2023

@author: tyrel
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:38:59 2023

@author: tyrel
"""
import tkinter as tk
import requests
import json

def on_predict_button_click():
    # Prepare the URL
    url = "http://127.0.0.1:12345/predict"

    # Prepare the JSON body from the user input
    data = {
        "BIKE_MODEL": bike_model_var.get(),
        "PRIMARY_OFFENCE": primary_offence_var.get(),
        "BIKE_MAKE": bike_make_var.get(),
        "LOCATION_TYPE": location_type_var.get(),
        "REPORT_DOY": int(report_doy_var.get()),  # Convert to int
        "OCC_DOY": int(occ_doy_var.get()),  # Convert to int
        "PREMISES_TYPE": premises_type_var.get(),
        "REPORT_HOUR": int(report_hour_var.get()),  # Convert to int
        "BIKE_SPEED": int(bike_speed_var.get()),  # Convert to int
        "BIKE_COST": float(bike_cost_var.get())  # Convert to float
    }

    # Send a GET request to the Flask server with JSON body
    response = requests.get(url, json=[data])  # Wrap data in a list

    # Display the server response
    response_text.delete(1.0, tk.END)  # Clear previous content
    response_text.insert(tk.END, response.text)


# Create the main window
root = tk.Tk()
root.title("Team 2 COMP309")

# Create and pack widgets
bike_model_var = tk.StringVar()
primary_offence_var = tk.StringVar()
bike_make_var = tk.StringVar()
location_type_var = tk.StringVar()
report_doy_var = tk.StringVar()
occ_doy_var = tk.StringVar()
premises_type_var = tk.StringVar()
report_hour_var = tk.StringVar()
bike_speed_var = tk.StringVar()
bike_cost_var = tk.StringVar()

label = tk.Label(root, text="Enter Bike Information:")
label.grid(row=0, column=0, columnspan=2, pady=10)

# Add labels and entry widgets for each field
fields = [
    "BIKE_MODEL", "PRIMARY_OFFENCE", "BIKE_MAKE", "LOCATION_TYPE",
    "REPORT_DOY", "OCC_DOY", "PREMISES_TYPE", "REPORT_HOUR",
    "BIKE_SPEED", "BIKE_COST"
]

for i, field in enumerate(fields):
    tk.Label(root, text=field + ":").grid(row=i + 1, column=0, sticky="e", padx=10)
    tk.Entry(root, textvariable=eval(field.lower() + "_var")).grid(row=i + 1, column=1, pady=5)

predict_button = tk.Button(root, text="Predict", command=on_predict_button_click)
predict_button.grid(row=len(fields) + 1, column=0, columnspan=2, pady=10)

response_text = tk.Text(root, height=5, width=40)
response_text.grid(row=len(fields) + 2, column=0, columnspan=2, pady=10)

# Start the main loop
root.mainloop()

