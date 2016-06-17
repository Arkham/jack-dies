import utils
import numpy as np
import pandas as pd
from sklearn import ensemble, cross_validation, grid_search

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print "\nCleaning up some data"

utils.clean_data(train)
utils.clean_data(test)

print "\nExtracting target and features"

print(train.shape)
target = train["Survived"].values
features = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

print "\nUse gradient boosting classifier"

# grid_search = grid_search.GridSearchCV(
#     estimator = ensemble.GradientBoostingClassifier(
#         learning_rate=0.001,
#         min_samples_split=40,
#         min_samples_leaf=1,
#         max_features=2,
#         max_depth=12,
#         n_estimators=70,
#         subsample=0.75,
#         random_state=10), 
#     param_grid = {'n_estimators':[140, 280, 560, 1120, 4480]},
#     scoring='roc_auc',
#     n_jobs=4,
#     iid=False,
#     cv=10)

# grid_search.fit(features, target)
# print(grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_)

gbm = ensemble.GradientBoostingClassifier(
    learning_rate = 0.005,
    min_samples_split=40,
    min_samples_leaf=1,
    max_features=2,
    max_depth=12,
    n_estimators=1500,
    subsample=0.75,
    random_state=1)
gbm = gbm.fit(features, target)

print(gbm.feature_importances_)
print(gbm.score(features, target))

# scores = cross_validation.cross_val_score(gbm, features, target, scoring='accuracy', cv=20)
# print scores
# print scores.mean()

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
prediction_gbm = gbm.predict(test_features)
utils.write_prediction(prediction_gbm, "results/gbm.csv")
