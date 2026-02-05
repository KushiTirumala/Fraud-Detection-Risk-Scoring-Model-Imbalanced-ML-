import joblib

def save_model(model, filename):
    """Save model to disk."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load model from disk."""
    return joblib.load(filename)
