# create_model.py
from sklearn.linear_model import LogisticRegression
import pickle

X = [[0, 1], [1, 0], [1, 1], [0, 0]]
y = [0, 1, 1, 0]

model = LogisticRegression()
model.fit(X, y)

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created in app/")
