import os
import urllib.request
import pandas as pd
import numpy as np

CLASS_NAMES = [
    "Normal",
    "Cyclic",
    "Increasing Trend",
    "Decreasing Trend",
    "Upward Shift",
    "Downward Shift"
]

def download_synthetic_control(dest_path="data/synthetic_control.data"):
    """Downloads the Synthetic Control dataset from UCI Repository if not present."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading Synthetic Control dataset from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print("Synthetic Control dataset already exists locally.")

def load_synthetic_control(file_path="data/synthetic_control.data"):
    """
    Loads the Synthetic Control dataset.
    Returns:
        X: pd.DataFrame of shape (600, 60)
        y: np.ndarray of shape (600,) containing class names
    """
    download_synthetic_control(file_path)
    
    # The file has space-separated floating point values
    # Let's read it with pandas
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    
    # Check shape
    if df.shape != (600, 60):
        # In case the separator is different
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        
    df.columns = [f"Time_{i+1}" for i in range(df.shape[1])]
    df.index = [f"Sample_{i+1}" for i in range(df.shape[0])]
    
    # Generate ground-truth labels (100 samples per class, in sequence)
    y = np.repeat(CLASS_NAMES, 100)
    
    return df, y

if __name__ == "__main__":
    X, y = load_synthetic_control()
    print("Data loaded successfully.")
    print("X shape:", X.shape)
    print("y unique classes:", np.unique(y, return_counts=True))
