from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    joblib.dump(clf, '/models/my_iris_model.pkl')
    while True:
       pass



