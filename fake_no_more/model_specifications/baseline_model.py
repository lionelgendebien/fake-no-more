from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

def initialize_model():
    model = DummyClassifier(strategy='most_frequent')
    return model

def train_model(model, X_train_scaled, y_train):
    return model.fit(X_train_scaled, y_train)

def compute_score(model, X_test_scaled,y_test):
    cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring='accuracy')
    return cv_scores.mean
