import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(solver='liblinear', max_iter=500)
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('logistic_regression', self.model)
        ])
        self.grid_params = {
            'logistic_regression__C': [0.01, 0.1, 1, 10],
            'logistic_regression__penalty': ['l1', 'l2']
        }
        self.grid_search = None

    def train(self, X, y):
        self.grid_search = GridSearchCV(self.pipeline, self.grid_params, cv=5, scoring='f1')
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_

    def predict(self, X):
        return self.best_model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        return metrics

class SVMModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(probability=True)
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('svm', self.model)
        ])
        self.grid_params = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf'],
            'svm__gamma': ['scale', 'auto']
        }
        self.grid_search = None

    def train(self, X, y):
        self.grid_search = GridSearchCV(self.pipeline, self.grid_params, cv=5, scoring='f1')
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_

    def predict(self, X):
        return self.best_model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        return metrics
