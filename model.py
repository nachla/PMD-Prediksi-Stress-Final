from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd

data = pd.read_csv(r"C:\Users\nachla\PMD\StressLevelDataset.csv")
X = data[['self_esteem', 'depression', 'anxiety_level', 'sleep_quality', 'bullying']]
y = data['stress_level']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
joblib.dump(rf_model, "rf_model.pkl")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)
joblib.dump(knn_model, "knn_model.pkl")
