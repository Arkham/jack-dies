import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.pyplot as plt
from sklearn import model_selection

# load data
df = pd.read_csv("data/train.csv")

# clear data
df = df.drop(['Ticket','Cabin'], axis=1)
df.Age = df.Age.interpolate()
df.Embarked = df.Embarked.fillna('S')

# run logistic regression
formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)"
results = {}

y,x = dmatrices(formula, data=df, return_type="dataframe")

model = sm.Logit(y,x)
res = model.fit()

results["Logit"] = [res, formula]
print(res.summary())

# print some stats
plt.figure(figsize=(18,4))

plt.subplot(121)
ypred = res.predict(x)
plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
plt.grid(color='white', linestyle='dashed')
plt.title('Logit predictions')

plt.subplot(122)
plt.plot(res.resid_dev, 'r-')
plt.grid(color='white', linestyle='dashed')
plt.title('Logit Residuals');

plt.show()
