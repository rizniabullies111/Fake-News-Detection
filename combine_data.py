import pandas as pd

df_fake = pd.read_csv('Fake.csv')
df_fake = df_fake.drop(['title','subject','date'], axis=1)
#df_fake.head()

df_fake = df_fake.assign(label='fake')
#df_fake.head()

df_true = pd.read_csv('True.csv')
df_true = df_true.drop(['title','subject','date'], axis=1)
#df_true.head()

df_true = df_true.assign(label='true')
#df_true.head()

df = pd.concat([df_true, df_fake])
df.to_csv('final_data.csv', index=False)