import pandas as pd
df=pd.read_csv("data/interim/preprocessed_train.csv")
print(df.sample(5))