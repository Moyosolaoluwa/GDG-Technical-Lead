import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# prepare features and target
X = df.drop(["Country", "Happiness score"], axis=1, errors="ignore")
y = df["Happiness score"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train random forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# save model
joblib.dump(rf, "final_model.pkl")
print("Final model saved as final_model.pkl")
