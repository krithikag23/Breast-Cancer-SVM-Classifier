import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 1 = benign, 0 = malignant
FEATURES = cancer.feature_names
TARGET_NAMES = cancer.target_names  # ['malignant', 'benign']

# Create SVM model with probability
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", probability=True, random_state=42)
)

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

def predict_sample(sample):
    """Input: dict of features → Output: prediction + confidence"""
    x = np.array(sample).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0][pred] * 100
    return TARGET_NAMES[pred], round(prob, 2)

def get_top_features(n=5):
    """Return the most influential features for the decision"""
    svm = model.named_steps["svc"]
    coef = svm.coef_[0]
    idx = np.argsort(np.abs(coef))[::-1][:n]
    return [(FEATURES[i], round(coef[i], 3)) for i in idx]
