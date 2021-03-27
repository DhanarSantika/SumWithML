import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sum(a, b, clf):
    predict = clf.predict(np.array([[a, b]], dtype=np.float32))
    result = predict[0]
    return result

clf = joblib.load('sum.pkl')
while True:
    x, y = input("Enter a two value: ").split()
    print(sum(x, y, clf))