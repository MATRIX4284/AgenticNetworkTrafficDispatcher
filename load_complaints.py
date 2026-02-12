import pandas as pd

def load_complaints(csv_path, limit=50):
    df = pd.read_csv(csv_path)
    return df[["Customer Complaint", "State"]].head(limit).to_dict("records")