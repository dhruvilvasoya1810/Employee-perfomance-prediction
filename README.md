# Employee-perfomance-prediction
Employee Performance Prediction is a Machine Learning project that uses Logistic Regression to predict employee performance into four categories: Poor, Average, Good, and Excellent. It uses Python, data preprocessing, EDA, and Supervised Learning, achieving 94.50% accuracy.

Employee Performance Prediction Using Machine Learning
A college mini project that uses supervised machine learning to classify employee performance into four categories: Poor, Average, Good, and Excellent.
Project Overview
This project analyzes employee-related information and trains a Logistic Regression classification model to predict an employee's performance category.

The dataset contains 1,000 employee records and includes details such as age, gender, department, education, experience, attendance, working hours, training hours, completed projects, overtime, salary, and job satisfaction.


Project Results
- Learning type: Supervised Learning
- Problem type: Multiclass Classification
- Algorithm: Logistic Regression
- Dataset size: 1,000 records
- Input features: 13
- Target classes: 4
- Reported test accuracy: 94.50%
The accuracy above is the result stated in the submitted project report. Results may vary if the dataset, preprocessing, or train/test split is changed.
Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- D-Tale
- Sweetviz
- ReportLab
- Joblib

- 
Repository Structure
project/
├── DATA SET/
│   └── employe.csv
├── Pythone code/
│   ├── employee.py
│   └── charts.py
└── project report/
    ├── project final report.pdf
    └── SWEETVIZ_REPORT.html

  
How to Run
1. Install Python on your computer.
2. Install the required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn dtale sweetviz reportlab joblib
3. Open Pythone code/employee.py in your Python editor.
4. Make sure the CSV file path in the script points to DATA SET/employe.csv on your computer. The current script contains a Windows-specific file path, so update it before running.
5. Run the script:
   python employee.py
The script trains the Logistic Regression model, prints evaluation results, displays a confusion matrix, and supports a sample prediction and an interactive user-input prediction.


Exploratory Data Analysis
The project includes data inspection and analysis such as:
- Dataset information and descriptive statistics
- Missing-value and unique-value checks
- Value counts for categorical columns
- Correlation and covariance analysis
- Sweetviz HTML report
- Data visualizations and confusion matrix 
Model Workflow
1. Load the employee CSV dataset.
2. Explore and preprocess the data.
3. Separate input features and the target column (Performance_Rating).
4. Split the data into training and testing sets (80:20).
5. Scale the input features using StandardScaler.
6. Train the Logistic Regression classifier.
7. Predict performance categories and evaluate the model.
8. Save the trained model and scaler as .pkl files when the script reaches the model-saving section.
9. 
Project Information
- Project: Employee Performance Prediction
- Course: SYBCA-AI, Semester III
- Subject: Fundamentals of Machine Learning (303-05)
- Project type: College Mini Project
- 
Author
Dhruvil G. Vasoya
