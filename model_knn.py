import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("Deepression.csv")
df.columns = df.columns.str.strip()

# Hapus baris yang memiliki NaN
df = df.dropna()

# Pisahkan fitur dan label
X = df.drop(columns=["Number", "Depression State"])
y = df["Depression State"]

# Encode label kategorikal
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Simpan model
joblib.dump(knn, "model.pkl")

print("Model KNN berhasil disimpan ke 'model.pkl'")
