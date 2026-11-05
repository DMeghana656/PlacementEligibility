import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = pd.read_csv("students.csv")

print("Student Dataset:")
print(data.head())

# 2. Separate input features and target
X = data[["CGPA", "Attendance", "SkillScore"]]
y = data["Eligible"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Train and compare entropy and Gini trees
models = {
    "Entropy (ID3-style)": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        random_state=42
    ),
    "Gini (CART-style)": DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Test accuracy:",
          round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 5. Choose the entropy model for visualization
model = models["Entropy (ID3-style)"]

plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Student Placement Eligibility - Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nDecision tree saved as decision_tree.png")

# 6. Predict for a new student
new_student = pd.DataFrame(
    [[8.2, 88, 82]],
    columns=["CGPA", "Attendance", "SkillScore"]
)

prediction = model.predict(new_student)[0]
probabilities = model.predict_proba(new_student)[0]

print("\nNew Student:")
print(new_student)

print("Predicted eligibility:", prediction)
print("Class probabilities:", dict(zip(model.classes_, probabilities)))
