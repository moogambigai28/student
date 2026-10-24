import math

import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a sample dataset of study hours and student marks.
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
df = pd.DataFrame(data)

# Display the dataset so it is clear what data the model will learn from.
print("Student Study Hours and Marks Dataset")
print(df.to_string(index=False))

# Separate the study hours (input) from the marks (value to predict).
X = df[["Study_Hours"]]
y = df["Marks"]

# Train a linear regression model using the sample data.
model = LinearRegression()
model.fit(X, y)

# Show the fitted line and how well it matches the training data.
print("\nModel Training Result")
print(f"Equation: marks = {model.coef_[0]:.2f} * study hours + {model.intercept_:.2f}")
print(f"Training R-squared score: {model.score(X, y):.2f}")

# Keep asking until the user enters a valid, non-negative number.
while True:
    try:
        hours = float(input("\nEnter the number of study hours: "))
        if not math.isfinite(hours) or hours < 0:
            print("Please enter a finite number that is zero or greater.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a number, such as 4 or 4.5.")

# Use the trained model to predict marks for the entered study hours.
new_data = pd.DataFrame({"Study_Hours": [hours]})
prediction = model.predict(new_data)[0]

# Display the prediction clearly.
print("\nPrediction")
print(f"Study hours: {hours:g}")
print(f"Expected marks: {prediction:.2f}")
