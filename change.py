import pandas as pd

df = pd.read_parquet("political_verified.parquet")
df.to_csv("political_verified.csv", index=False)