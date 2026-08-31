import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\mlops_day1\\data\\data.csv")


# Features and target
X = df.drop(columns=["sales", "Unnamed: 0"])
y = df["sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=67
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Prediction
predictions = model.predict(X_test)


# Evaluation
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("RMSE:", rmse)
print("R2 Score:", r2)


# Save model
joblib.dump(model, "models/linear_reg_model.pkl")


print("Model saved successfully!")