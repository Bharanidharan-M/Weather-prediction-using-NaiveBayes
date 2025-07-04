# Weather prediction using NaiveBayes
# Australian Weather Dataset - Rain Prediction with Naive Bayes

This project performs preprocessing, encoding, and classification on the **Australian Weather Dataset** to predict whether it will rain tomorrow (`RainTomorrow`). The model used is **Gaussian Naive Bayes**.

## 📂 Dataset

- **File:** `weatherAUS.xlsx`
- **Features:** Weather observations like temperature, humidity, wind, pressure, cloud cover, etc.
- **Target:** `RainTomorrow` (0 = No, 1 = Yes)

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`

## ⚙️ Workflow

✅ **Data Loading**  
- Read dataset using `pandas.read_excel`.

✅ **Missing Value Handling**  
- Categorical columns filled with mode.  
- Numerical columns filled with median.

✅ **Label Encoding**  
- Convert categorical columns to numeric using `LabelEncoder`.

✅ **Train/Test Split**  
- 80% train, 20% test using `train_test_split`.

✅ **Model**  
- `GaussianNB` from `sklearn.naive_bayes`.

✅ **Evaluation**  
- Accuracy score  
- Confusion matrix  

## 📊 Results

- **Accuracy:** 80.11%  
- **Confusion Matrix:**  
  ```
  [[19448  3224]
   [ 2561  3859]]
  ```

## 🚀 How to Run

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = pd.read_excel('/content/weatherAUS.xlsx')

# Fill missing values
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtypes == 'object':
        data[col] = le.fit_transform(data[col])

# Split data
x = data.drop(['RainTomorrow', 'Date'], axis=1)
y = data['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion_matrix:", confusion_matrix(y_test, y_pred))
```

## 📎 References

- [Bureau of Meteorology (Australia) dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

✅ *Tip:* View this notebook on [nbviewer](https://nbviewer.org/) for better rendering of outputs on GitHub.

