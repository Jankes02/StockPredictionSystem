import pandas as pd
import glob

files = glob.glob("./data/daily/*.csv")

cutoff = "2026-03-31"

for file in files:
    df = pd.read_csv(file, parse_dates=["Date"])
    
    # filter rows
    df = df[df["Date"] <= cutoff]
    
    # overwrite (or save elsewhere)
    df.to_csv(file, index=False)