import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
import numpy as np

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create average score and Pass/Fail label
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Encode categorical columns
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education',
                    'lunch', 'test preparation course', 'result']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# =========================
# 1. Classification with Random Forest
# =========================

# Separate Pass/Fail classes
pass_class = df[df['result'] == label_encoders['result'].transform(['Pass'])[0]]
fail_class = df[df['result'] == label_encoders['result'].transform(['Fail'])[0]]

# Balance dataset (downsample majority)
pass_downsampled = resample(pass_class,
                            replace=False,
                            n_samples=len(fail_class),
                            random_state=42)

df_balanced = pd.concat([pass_downsampled, fail_class]).sample(frac=1, random_state=42)

# Features and target for classification
X_class = df_balanced[['gender', 'race/ethnicity', 'parental level of education',
                       'lunch', 'test preparation course']]
y_class = df_balanced['result']

# Train/test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_c, y_train_c)

# Save classification model
with open("student_performance_rf_classifier.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)

# Classification report
from sklearn.metrics import classification_report
y_pred_c = rf_classifier.predict(X_test_c)
print("\n=== Random Forest Classifier Report ===")
print(classification_report(y_test_c, y_pred_c))

# =========================
# 2. Regression with Linear Regression (Predict average_score)
# =========================

# Features and target for regression
X_reg = df[['gender', 'race/ethnicity', 'parental level of education',
            'lunch', 'test preparation course']]
y_reg = df['average_score']

# Train/test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)

# Save regression model
with open("student_performance_linear_reg.pkl", "wb") as f:
    pickle.dump(lin_reg, f)

# Evaluate regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred_r = lin_reg.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)

print("\n=== Linear Regression Performance (Average Score Prediction) ===")
print(f"MAE: {mean_absolute_error(y_test_r, y_pred_r):.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2_score(y_test_r, y_pred_r):.2f}")