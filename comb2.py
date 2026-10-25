import pandas as pd

df = pd.read_parquet("kenyan_news_political_classified.parquet")
df1 = pd.read_parquet("kenyan_news_political_classified1.parquet")

# Keep and standardize the four columns
df = df[['url', 'website', 'article_text', 'political_classification']]

df1 = df1.rename(columns={
    'news_company': 'website',
    'article': 'article_text'
})

df1 = df1[['url', 'website', 'article_text', 'political_classification']]

# Combine row-wise
data = pd.concat([df, df1], ignore_index=True)

print(data.shape)
print(data.columns)


data.to_parquet('data.parquet')
print(data['political_classification'].value_counts())

political=data[data['political_classification']=='Political']

print(political.shape)
political.to_parquet('political.parquet')