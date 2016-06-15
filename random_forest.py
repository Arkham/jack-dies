import utils
import numpy as np
import pandas as pd
from sklearn import ensemble

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print "\nCleaning up some data"

utils.clean_data(train)
utils.clean_data(test)

print "\nExtracting target and features"

print(train.shape)
target = train["Survived"].values
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

print "\nUse Random Forest classifier"

forest = ensemble.RandomForestClassifier(
    max_depth = 7,
    min_samples_split = 2,
    n_estimators = 10000,
    random_state = 1,
    n_jobs = -1
)
forest = forest.fit(features_forest, target)

print(forest.feature_importances_)
print(forest.score(features_forest, target))

test_features_forest = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_forest = forest.predict(test_features_forest)
utils.write_prediction(prediction_forest, "results/random_forest.csv")
