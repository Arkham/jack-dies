import utils
import numpy as np
import pandas as pd
from sklearn import tree, cross_validation

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print "\nCleaning up some data"

utils.clean_data(train)
utils.clean_data(test)

print "\nExtracting target and features"

print(train.shape)
target = train["Survived"].values
features = train[["Pclass", "Sex", "Age", "Fare"]].values

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree = decision_tree.fit(features, target)

print(decision_tree.feature_importances_)
print(decision_tree.score(features, target))

print "\nTry on test set"

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
prediction = decision_tree.predict(test_features)
utils.write_prediction(prediction, "results/decision_tree.csv")

print "\nCorrect overfitting"

feature_names = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
features_two = train[feature_names].values
decision_tree_two = tree.DecisionTreeClassifier(
    max_depth = 7,
    min_samples_split = 2,
    random_state = 1)
decision_tree_two = decision_tree_two.fit(features_two, target)

print(decision_tree_two.feature_importances_)
print(decision_tree_two.score(features_two, target))
tree.export_graphviz(decision_tree_two, feature_names=feature_names, out_file="./graphs/decision_tree_two.dot")

scores = cross_validation.cross_val_score(decision_tree_two, features_two, target, scoring='accuracy', cv=10)
print scores
print scores.mean()

print "\nWrite new predicition"

test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_two = decision_tree_two.predict(test_features_two)
utils.write_prediction(prediction_two, "results/decision_tree_two.csv")

