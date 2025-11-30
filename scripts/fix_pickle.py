import pickle
import pathlib
import sys

from emhass.utils import (
    get_root,
)

# Define the path to test data file
root = pathlib.Path(str(get_root(__file__, num_parent=2)))
file_path = root / "data/test_df_final.pkl"

if not file_path.exists():
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

print(f"Migrating {file_path} to newer NumPy version...")

# 1. Load the data (This will trigger the warning one last time)
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 2. Save the data back (This writes it using the new NumPy structure)
with open(file_path, "wb") as f:
    pickle.dump(data, f)

print("Done! The pickle file has been updated.")