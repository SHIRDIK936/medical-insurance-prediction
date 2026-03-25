import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# 1. Load and Preprocess
df = pd.read_csv("insurance.csv")
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Build Deep Learning Model (ANN)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear') # Linear for regression (price prediction)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=0)

# 5. Save Model and Scaler
# We save the model as a .pkl using a wrapper or .h5/keras format. 
# To keep your app.py simple, we'll save the weights via pickle.
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Deep Learning model.pkl and scaler.pkl created successfully")