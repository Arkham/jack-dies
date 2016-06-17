import pandas as pd

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print "Survived / passed counts"
print(train["Survived"].value_counts())

print "\nSurvived / passed as percentage"
print(train["Survived"].value_counts(normalize = True))

print "\nMales survived / males passed"
print(train["Survived"][train["Sex"] == "male"].value_counts(normalize = True))

print "\nFemals survived / females passed"
print(train["Survived"][train["Sex"] == "female"].value_counts(normalize = True))

print "\nAdding a new Child column"
train["Child"] = float('NaN')
train.loc[train["Age"] >= 18, "Child"] = 0
train.loc[train["Age"] < 18, "Child"] = 1

print "\nChildren survived / children passed"
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

print "\nPredict 1 if female and 0 if male"
test_one = test
test_one["Survived"] = 0
test_one.loc[test_one["Sex"] == 'female', "Survived"] = 1
test_one.to_csv("results/gender_model.csv", index = False, columns = ["PassengerId", "Survived"])

print "\nCheck how accurate this model is on training set"
train["Hyp"] = 0
train.loc[train["Sex"] == "female", "Hyp"] = 1

train["Result"] = 0
train.loc[train["Hyp"] == train["Survived"], "Result"] = 1
print(train["Result"].value_counts(normalize = True))
