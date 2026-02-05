from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def feature_selection(X, y, k=10):
    """Select top k features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

def train_model(X_train, y_train, model_type='RandomForest'):
    """Train a classifier."""
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate classifier."""
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
