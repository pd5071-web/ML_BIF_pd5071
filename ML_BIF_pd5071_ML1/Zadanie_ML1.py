# pd5071
# Uczenie maszynowe w bioinformatyce
# Zadanie ML1

import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("dane_projekt1.csv")

# Statystyki
print(df.describe())

# Podział na cechy i etykiety
X = df.drop(columns=["Gene_Function"])
y = df["Gene_Function"]

# Train + test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,
    stratify=y_temp,
    random_state=42
)

# Rozmiary zbiorów
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
