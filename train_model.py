from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model to app directory
joblib.dump(model, "app/model.joblib")

print("✅ Model saved to app/model.joblib")
