import pandas as pd

def load_medicine_db():
    df = pd.read_csv("dataset/medicines.csv")   # put correct filename

    # Take only medicine names
    medicines = df["name"].dropna().unique().tolist()

    # Clean names (important)
    medicines = [m.strip().lower() for m in medicines]

    return medicines

medicine_list = load_medicine_db()