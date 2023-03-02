import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("hr.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.map(lambda x: 1 if x == 'Canceled' else 0)

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
