
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# --- 1. DATA GENERATION (Improved with distinct profiles) ---
np.random.seed(42) # for reproducibility

data_size = 300
half_size = data_size // 2

# Generate 'No Churn' data: very high tenure, very low monthly charges
no_churn_data = {
    'tenure': np.random.randint(48, 72, size=half_size), # High tenure (4-6 years)
    'monthly_charges': np.random.uniform(20, 40, size=half_size), # Low charges
    'total_charges': np.random.uniform(1000, 5000, size=half_size),
    'churn': np.zeros(half_size, dtype=int)
}

# Generate 'Churn' data: very low tenure, very high monthly charges
churn_data = {
    'tenure': np.random.randint(1, 12, size=half_size), # Low tenure (less than 1 year)
    'monthly_charges': np.random.uniform(90, 120, size=half_size), # High charges
    'total_charges': np.random.uniform(50, 1000, size=half_size),
    'churn': np.ones(half_size, dtype=int)
}

# Combine the two datasets
df_no_churn = pd.DataFrame(no_churn_data)
df_churn = pd.DataFrame(churn_data)
df = pd.concat([df_no_churn, df_churn], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# --- 2. PREPARE DATA FOR MODELING ---
X = df[['tenure', 'monthly_charges', 'total_charges']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN THE MODEL ---
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# --- 4. EVALUATE AND SAVE ---
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

print("✅ Trained model saved as 'churn_model.pkl'")
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

joblib.dump(model, 'churn_model.pkl')
