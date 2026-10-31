import pandas as pd
import numpy as np
import dtale as d
import sweetviz as s
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

#Data set

df=pd.read_csv("C:\\Users\\ventura\\Desktop\\project\\DATA SET\\employe.csv")
df.info()

#numeric data

print("\nquantitative data:int64 and float64")
quantitative=df.select_dtypes(include=['int64','float64'])
print("quantitative data:",quantitative)
print("quantitative column labels:\n",quantitative.columns)

#catrgorycal data

print("\n qualitative data:object and category")
qualitative=df.select_dtypes(include=['object','category'])
print("Qualitative data:",qualitative)
print("qualitative column labels:\n",qualitative.columns)

#EDA

print(df.head(5))
print(df.tail(5))
print(df.isnull().sum())
print(df.describe())
print(df.nunique())

#covariation and corelation
print(df.cov(numeric_only=True))
print(df.corr(numeric_only=True))

#value count

print(df['Gender'].value_counts())
print(df['Department'].value_counts())
print(df['Education'].value_counts())
print(df['Performance_Rating'].value_counts())
print(df['Overtime'].value_counts())

#Data preprocessing

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


print(df.head(5))
print(df.tail(6))

target_encoder= LabelEncoder()
df["Performance_Rating"]=target_encoder.fit_transform(df["Performance_Rating"])

print(df.head(5))

# Drop columns


x=df.drop(columns=['Employee_ID','Performance_Rating'])
y=df['Performance_Rating']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LogisticRegression(max_iter=1000)
model.fit(x_train_scaled,y_train)

y_pred=model.predict(x_test_scaled)

'''#R2 and RSS

r2=model.score(x_test_scaled,y_test)
print("r2 value:",r2)

rss = ((y_test-y_pred)**2).sum()
print("RSS:",rss)'''

#ACCURACY CHECK

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

print(classification_report(y_test,y_pred))

'''#MAE & MSE

mae=mean_absolute_error(y_test,y_pred)
print("mean mae:",mae)

mse=mean_squared_error(y_test,y_pred)
print("mean mse:",mse)'''

cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
pt.show()



# ==========================================
# VISUALIZATION
# ==========================================

'''# 1. Performance Rating Distribution
pt.figure(figsize=(8,5))
sns.countplot(x=df["Performance_Rating"])
pt.title("Performance Rating Distribution")
pt.xlabel("Performance Rating")
pt.ylabel("Number of Employees")
pt.show()


# 2. Gender Distribution
pt.figure(figsize=(6,4))
sns.countplot(x=df["Gender"])
pt.title("Gender Distribution")
pt.xlabel("Gender")
pt.ylabel("Number of Employees")
pt.show()


# 3. Department Distribution
pt.figure(figsize=(8,5))
sns.countplot(x=df["Department"])
pt.title("Department Distribution")
pt.xlabel("Department")
pt.ylabel("Number of Employees")
pt.xticks(rotation=45)
pt.show()


# 4. Overtime vs Performance
pt.figure(figsize=(7,5))
sns.countplot(x=df["Overtime"], hue=df["Performance_Rating"])
pt.title("Overtime vs Performance Rating")
pt.xlabel("Overtime")
pt.ylabel("Number of Employees")
pt.show()


# 5. Education vs Performance
pt.figure(figsize=(8,5))
sns.countplot(x=df["Education"], hue=df["Performance_Rating"])
pt.title("Education vs Performance Rating")
pt.xlabel("Education")
pt.ylabel("Number of Employees")
pt.xticks(rotation=45)
pt.show()

# CORRELATION HEATMAP

pt.figure(figsize=(12,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

pt.title("Correlation Heatmap")
pt.show()

# ACTUAL VS PREDICTED


pt.figure(figsize=(8,5))

pt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual"
)

pt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted"
)

pt.title("Actual vs Predicted Performance")
pt.xlabel("Employee")
pt.ylabel("Performance Rating")
pt.legend()

pt.show()


# CONFUSION MATRIX WITH LABELS

cm = confusion_matrix(y_test, y_pred)

pt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

pt.title("Confusion Matrix")
pt.xlabel("Predicted")
pt.ylabel("Actual")

pt.show()

# MODEL ACCURACY

print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print("Accuracy :", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("Sweetviz report generated successfully!")'''

# ==========================================
# SAMPLE PREDICTION
# ==========================================

print("\n===================================")
print("       SAMPLE PREDICTION")
print("===================================")

sample = x_test.iloc[[0]]

prediction = model.predict(
    scaler.transform(sample)
)

print("Predicted Performance Rating:", prediction[0])
# ==========================================
# SAVE MODEL
# ==========================================

import joblib

joblib.dump(model, "employee_performance_model.pkl")
joblib.dump(scaler, "employee_scaler.pkl")

print("\nModel saved successfully!")
print("employee_performance_model.pkl")
print("employee_scaler.pkl")
# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n===================================")
print(" EMPLOYEE PERFORMANCE PREDICTION")
print("          PROJECT COMPLETED")
print("===================================")

# ==========================================
# USER INPUT PREDICTION SYSTEM
# ==========================================

print("\n")
print("==========================================")
print("   EMPLOYEE PERFORMANCE PREDICTION")
print("==========================================")

# Read original dataset again
original_df = pd.read_csv(
    r"C:\Users\ventura\Desktop\project\DATA SET\employe.csv"
)

# Create encoders for categorical columns
encoders = {}

for col in original_df.select_dtypes(include=['object']).columns:

    if col != "Performance_Rating":

        encoder = LabelEncoder()

        encoder.fit(original_df[col].astype(str))

        encoders[col] = encoder


# Target encoder
target_encoder = LabelEncoder()

target_encoder.fit(
    original_df["Performance_Rating"].astype(str))
# ------------------------------------------
# TAKE USER INPUT
# ------------------------------------------

user_data = {}

print("\nEnter Employee Details:\n")

for col in x.columns:

    # Employee_ID is already removed from X,
    # so only model input columns will be asked.

    if col in encoders:

        print("\nAvailable", col, "values:")

        print(
            list(encoders[col].classes_)
        )

        value = input(
            "Enter " + col + ": "
        )

        # Check valid category
        while value not in encoders[col].classes_:

            print("Invalid value!")

            print(
                "Please choose from:",
                list(encoders[col].classes_)
            )

            value = input(
                "Enter " + col + ": "
            )

        user_data[col] = encoders[col].transform(
            [value]
        )[0]

    else:

        value = float(
            input(
                "Enter " + col + ": "
            )
        )

        user_data[col] = value

# ------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------

user_df = pd.DataFrame(
    [user_data],
    columns=x.columns
)
print("\nUser Input Data:")
print(user_df)

# ------------------------------------------
# SCALE USER DATA
# ------------------------------------------

user_scaled = scaler.transform(
    user_df)

# ------------------------------------------
# PREDICTION
# ------------------------------------------

prediction = model.predict(
    user_scaled)

# ------------------------------------------
# CONVERT PREDICTION TO ORIGINAL LABEL
# ------------------------------------------

predicted_rating = target_encoder.inverse_transform(
    prediction.astype(int)
)[0]
print("\n")
print("==========================================")
print("       PREDICTION RESULT")
print("==========================================")
print(
    "Predicted Employee Performance:",
    predicted_rating
)
print("==========================================")
# Python source file
python_file = "employee.py"

# PDF file name
pdf_file = "Employee_Performance_Project_Code.pdf"

# Create PDF
c = canvas.Canvas(pdf_file, pagesize=A4)

width, height = A4
# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(50, height - 50, "Employee Performance Prediction")
c.setFont("Helvetica", 11)
c.drawString(50, height - 70, "Complete Python Source Code")

# Read Python code
with open(python_file, "r", encoding="utf-8") as file:
    code = file.readlines()

# Code settings
c.setFont("Courier", 7)

x = 40
y = height - 100
line_height = 9

# Write code into PDF
for line in code:

    # New page if space is finished
    if y < 40:
        c.showPage()
        c.setFont("Courier", 7)
        y = height - 40

    # Remove unsupported characters
    line = line.rstrip()

    # Break long lines
    while len(line) > 100:
        part = line[:100]
        c.drawString(x, y, part)
        y -= line_height

        if y < 40:
            c.showPage()
            c.setFont("Courier", 7)
            y = height - 40

        line = line[100:]

    c.drawString(x, y, line)
    y -= line_height

# Save PDF
c.save()

print("====================================")
print("PDF CREATED SUCCESSFULLY")
print("====================================")
print("File:", pdf_file)
print("====================================")