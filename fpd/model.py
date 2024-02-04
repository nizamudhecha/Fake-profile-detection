import joblib
from sklearn.ensemble import RandomForestClassifier  # Replace with your actual model

# Create and train your machine learning model (Replace this with your actual model)
model = RandomForestClassifier(n_estimators=100)
# Train your model with data

# Save the trained model to a file
model_file = 'model.pkl'
joblib.dump(model, model_file)

print(f"Model saved as '{model_file}'.")