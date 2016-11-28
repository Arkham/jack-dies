import utils
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print "\nCleaning up some data"

utils.clean_data(train)
utils.clean_data(test)

print "\nExtracting target and features"

print(train.shape)
target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

print "\nUse logistic regression"

logistic = linear_model.LogisticRegression()
logistic.fit(features, target)
print(logistic.score(features, target))

scores = model_selection.cross_val_score(logistic, features, target, scoring='accuracy', cv=10)
print scores
print scores.mean()

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
utils.write_prediction(logistic.predict(test_features), "results/logistic_regression.csv")

print "\nUse polynomial features"
poly = preprocessing.PolynomialFeatures(degree=2)
features_ = poly.fit_transform(features)

clf = linear_model.LogisticRegression(C=10)
clf.fit(features_, target)
print(clf.score(features_, target))

scores = model_selection.cross_val_score(clf, features_, target, scoring='accuracy', cv=10)
print scores
print scores.mean()

test_features_ = poly.fit_transform(test_features)
utils.write_prediction(clf.predict(test_features_), "results/logistic_regression_poly.csv")
