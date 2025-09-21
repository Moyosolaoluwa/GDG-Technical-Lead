import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Stage 1: Data Cleaning

df = pd.read_csv("WHR_2023.csv")
print(df.head())
print(df.info())

# remove duplicates
df = df.drop_duplicates()

# fill missing values with average
df = df.fillna(df.mean(numeric_only=True))

# save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("Cleaned data saved as cleaned_data.csv")

# Stage 1: Visualization

# happiness vs GDP
sns.scatterplot(x="GDP per capita", y="Happiness score", data=df)
plt.show()

# top 10 happiest countries
top10 = df.nlargest(10, "Happiness score")
sns.barplot(x="Happiness score", y="Country", data=top10)
plt.show()

# correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# Stage 2: Machine Learning

X = df.drop(["Country", "Happiness score"], axis=1, errors="ignore")
y = df["Happiness score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model 1: linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear R2:", r2_score(y_test, lr.predict(X_test)))

# model 2: random forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print("Random Forest R2:", r2_score(y_test, rf.predict(X_test)))

# pick best model
best_model = rf
joblib.dump(best_model, "final_model.pkl")
print("Saved best model as final_model.pkl")
