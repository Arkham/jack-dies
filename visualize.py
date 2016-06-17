import matplotlib.pyplot as plt
import pandas as pd

# read csv
df = pd.read_csv("./data/train.csv")

# configure graph
fig = plt.figure(figsize=(18,6))
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.55

# survived vs deceased
plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Distribution of Survival, (1 = Survived)")

# survival by age
plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
plt.ylabel("Age")
plt.grid(b=True, which="major", axis="y")
plt.title("Survival by Age, (1 = Survived)")

# class distribution
plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
plt.title("Class Distribution")

# age distribution within class
plt.subplot2grid((2,3),(1,0), colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.xlabel("Age")
plt.title("Age Distribution within Classes")
plt.legend(("1st Class", "2nd Class", "3rd Class"))

# passengers boarding location
plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Passengers boarding location")

plt.show()
