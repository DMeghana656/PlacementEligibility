# Student Placement Eligibility Predictor

A small Python project for exploring student placement eligibility with a decision tree. It includes a Streamlit dashboard for interactive predictions and dataset management, plus a command-line script that trains and evaluates decision trees from the sample CSV.

> **Educational demo only:** The eligibility result follows the project’s fixed criteria and is not an official placement decision or a guarantee of placement.

## Eligibility criteria

A student is marked eligible only when **all three** conditions are met:

| Measure | Requirement |
| --- | ---: |
| CGPA (out of 10) | `>= 6.5` |
| Attendance (percent) | `>= 45` |
| Skill score (out of 100) | `>= 60` |

## Features

- **Dashboard:** View total, eligible, and not-eligible student counts and enter details for an interactive prediction.
- **Student Dataset:** View and edit the sample data, add or delete rows, and save changes to `students.csv`.
- **Decision Tree:** Explore a visualization of the dashboard's decision rules.
- **Command-line evaluation:** Compare entropy- and Gini-based decision trees and save a tree visualization as `decision_tree.png`.

## Dataset format

`students.csv` contains these columns:

| Column | Description |
| --- | --- |
| `RollNumber` | Student identifier; must be unique when editing data in the app. |
| `StudentName` | Student name. |
| `CGPA` | CGPA on a 0–10 scale. |
| `Attendance` | Attendance percentage on a 0–100 scale. |
| `SkillScore` | Skill score on a 0–100 scale. |
| `Eligible` | Eligibility label (`Yes` or `No`). The app recalculates this field from the criteria above. |

