import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

Y = data['meal']
# make sure you drop a column with the axis=1 argument
columns_to_drop = ['id', 'meal', 'DateTime']
X = data.drop(columns=columns_to_drop)


x, xt, y, yt = train_test_split(X, Y, test_size=0.2)

model = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.4, objective='binary:hinge')

y = y.to_frame()

modelFit = model.fit(x, y)

pred1 = model.predict(xt)
pred = pred1.flatten().tolist()

print(accuracy_score(yt, pred)*100)