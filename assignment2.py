import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

Y = data['meal']
# make sure you drop a column with the axis=1 argument
columns_to_drop = ['id', 'DateTime', 'meal']
X = data.drop(columns=columns_to_drop, axis=1)



x, xt, y, yt = train_test_split(X, Y, test_size=1000, random_state=67) 

from sklearn.ensemble import RandomForestClassifier

# Generate the random forest model
model = RandomForestClassifier(n_estimators=1500, 
                               max_depth=25, 
                               n_jobs = -1,
                              min_samples_leaf=2,
                              min_samples_split=4,
                            	random_state=67)
# Fit the model to the training data
modelFit = model.fit(x, y)
# Make predictions
pred = model.predict(xt)
# Print the accuracy score of the fitted model
print("The random forest has an accuracy of : %s\n" 
	% str(accuracy_score(pred, yt)))