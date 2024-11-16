import pandas as pd
import numpy as np
from models import LogisticRegressionModel, SVMModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def process_dataset():
    print("Loading dataset.....")
    ecom_data = pd.read_csv('../data/e_commerce.csv')
    print("Dataset loaded!")
    print("Pre-processing.....")
    ecom_data.drop_duplicates(subset=['user_id'], inplace=True)
    ecom_data = ecom_data.drop(
        ecom_data.query('(group == "treatment" and landing_page != "new_page") or \
                         (group == "control" and landing_page != "old_page")').index
    )
    ecom_data['group'] = LabelEncoder().fit_transform(ecom_data['group'])
    ecom_data['landing_page'] = LabelEncoder().fit_transform(ecom_data['landing_page'])
    X = ecom_data[['group', 'landing_page']]
    y = ecom_data['converted']
    print("Pre-processing finished!")
    return X, y

X, y = process_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training and testing models...")
logistic_model = LogisticRegressionModel()
logistic_model.train(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

svm_model = SVMModel()
svm_model.train(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print("Training and testing finished!\n")

print("Logistic Regression Predictions:", logistic_predictions[:10])
print("SVM Predictions:", svm_predictions[:10])