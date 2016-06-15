import numpy as np
from sklearn import tree, ensemble
import pandas as pd

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

print "\nCleaning up some data"

clean_data(train)
clean_data(test)

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

PassengerId = np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
print(solution.shape)
solution.to_csv("decision_tree.csv", index_label = ["PassengerId"])

print "\nCorrect overfitting"

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
decision_tree_two = tree.DecisionTreeClassifier(
    max_depth = 10,
    min_samples_split = 5,
    random_state = 1)
decision_tree_two = decision_tree_two.fit(features_two, target)

print(decision_tree_two.feature_importances_)
print(decision_tree_two.score(features_two, target))

print "\nWrite new predicition"

test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_two = decision_tree_two.predict(test_features_two)

PassengerId = np.array(test["PassengerId"]).astype(int)
solution_two = pd.DataFrame(prediction_two, PassengerId, columns = ["Survived"])
print(solution_two.shape)
solution_two.to_csv("decision_tree_two.csv", index_label = ["PassengerId"])

print "\nUse Random Forest classifier"

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
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

PassengerId = np.array(test["PassengerId"]).astype(int)
solution_forest = pd.DataFrame(prediction_forest, PassengerId, columns = ["Survived"])
print(solution_forest.shape)
solution_forest.to_csv("decision_tree_forest.csv", index_label = ["PassengerId"])
