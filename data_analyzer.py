import pandas as pd

# Load the dataset
df = pd.read_parquet('political_verified.parquet')

# Basic statistics
print("Total number of articles:", len(df))
print("\nArticles per website:")
print(df['website'].value_counts())

# If classification exists, show distribution
if 'political_classification' in df.columns:
    print("\nPolitical classification distribution:")
    print(df['political_classification'].value_counts())

if 'judge_independent_label' in df.columns:
    print("\nLabel distribution:")
    print(df['judge_independent_label'].value_counts())
