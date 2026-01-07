import os
import webbrowser

import numpy as np
import pandas as pd

from emhass import utils


def generate_debug_html():
    # Create Dummy Data (Same as the test)
    generator = np.random.default_rng()
    dates = pd.date_range(start="2024-01-01", periods=48, freq="30min")
    df = pd.DataFrame(index=dates)
    df["P_PV"] = generator.standard_normal(48) * 4000
    df["P_Load"] = generator.standard_normal(48) * 2000
    df["P_grid"] = df["P_Load"] - df["P_PV"]
    df["optim_status"] = "Optimal"
    df["cost_fun_profit"] = -2.5
    df["unit_load_cost"] = 0.20

    # Add Thermal Data (The new feature)
    df["predicted_temp_heater1"] = 21.0 + generator.standard_normal(48) * 2
    df["target_temp_heater1"] = 22.0

    # Get the Injection Dictionary
    print("Generating plots...")
    injection_dict = utils.get_injection_dict(df.copy())

    # Write to HTML file
    filename = "debug_plots.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visual Test of EMHASS Plots</title>
            <style>
                body { font-family: sans-serif; padding: 20px; text-align: center; }
                .table_div { margin-bottom: 40px; border: 1px solid #ddd; padding: 10px; }
                table { margin: 0 auto; border-collapse: collapse; }
                th, td { padding: 8px; border: 1px solid #ccc; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Debug Visualization</h1>
        """)

        # Loop through the dictionary items just like index.html does
        # Note: In Python 3.7+, dict insertion order is preserved,
        # so they will appear in the order defined in utils.py
        for _, content in injection_dict.items():
            f.write('<div class="table_div">\n')
            f.write(content)
            f.write("</div>\n")

        f.write("</body></html>")

    print(f"✅ Successfully created {filename}")

    # Automatically open in browser
    webbrowser.open("file://" + os.path.realpath(filename))


if __name__ == "__main__":
    generate_debug_html()
