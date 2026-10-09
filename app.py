from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Find the CSV file in the project folder
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "students.csv"

# Load the student dataset
df = pd.read_csv(CSV_PATH)

# Prepare the input features and target
features = ["StudyHours", "Attendance", "PreviousMarks"]
X = df[features]
y = df["Result"].str.strip().str.title()

# Train the machine learning model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    student = {}

    if request.method == "POST":
        try:
            hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            marks = float(request.form["previous_marks"])

            student = {
                "study_hours": hours,
                "attendance": attendance,
                "previous_marks": marks
            }

            if not 0 <= hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= attendance <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= marks <= 100:
                raise ValueError("Marks must be between 0 and 100.")

            new_student = pd.DataFrame(
                [[hours, attendance, marks]],
                columns=features
            )

            prediction = model.predict(new_student)[0]

        except ValueError as exc:
            error = str(exc) or "Please enter valid numbers."

        except (KeyError, TypeError):
            error = "Please fill in all student details correctly."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        student=student
    )


if __name__ == "__main__":
    app.run(debug=True)