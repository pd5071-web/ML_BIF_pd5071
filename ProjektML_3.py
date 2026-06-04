# pd5071
# Uczenie maszynowe w bioinformatyce
# Projekt 3: Przewidywanie wpływu mutacji DNA na poziom ekspresji genu

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("projekt_dane3.csv")

# Eksploracja danych

print("Pierwsze 5 rekordów:")
print(df.head())
print("\nInformacje o danych:")
print(df.info())
print("\nPodstawowe statystyki:")
print(df.describe())
print("\nBrakujące wartości:")
print(df.isnull().sum())

# Przetwarzanie danych
#   kategorie

from sklearn.preprocessing import LabelEncoder

le_gene = LabelEncoder()

df["Gene"] = le_gene.fit_transform(df["Gene"])

# Kodowanie nukleotydów

le_ref = LabelEncoder()
le_alt = LabelEncoder()

df["Reference_Base"] = le_ref.fit_transform(df["Reference_Base"])
df["Alternate_Base"] = le_alt.fit_transform(df["Alternate_Base"])

df["Sequence_Length"] = df["DNA_Sequence"].apply(len)

print("\nUnikalne długości sekwencji:")
print(df["Sequence_Length"].unique())

df.drop("DNA_Sequence", axis=1, inplace=True)

# Podział danych

from sklearn.model_selection import train_test_split

X = df.drop("Expression_Level", axis=1)

y = df["Expression_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Budowa modelu

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Ewaluacja modelu

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n===== WYNIKI MODELU =====")

# MAE

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", round(mae, 2))

# RMSE

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

print("RMSE:", round(rmse, 2))

# R²

from sklearn.metrics import r2_score

r2 = r2_score(
    y_test,
    y_pred
)

print("R²:", round(r2, 4))

# Współczynniki regresji

coefficients = pd.DataFrame({
    "Cecha": X.columns,
    "Współczynnik": model.coef_
})

print("\nWpływ cech na model:")
print(coefficients.sort_values(
    by="Współczynnik",
    key=abs,
    ascending=False
))

# Wizualizacja

plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Rzeczywiste wartości")
plt.ylabel("Przewidziane wartości")
plt.title("Regresja liniowa - rzeczywiste vs przewidziane")

plt.tight_layout()
plt.show()

"""
# Wnioski
 Zastosowano regresję liniową do przewidywania poziomu ekspresji genu na podstawie cech mutacji DNA. 
 Uzyskane wyniki (MAE = 40.45, RMSE = 46.04, R² = -0.168) wskazują jednak na niską skuteczność modelu. 
 Ujemna wartość R² oznacza, że model przewiduje gorzej niż proste wykorzystanie średniej wartości ekspresji. 
 Prawdopodobną przyczyną jest brak silnych zależności liniowych oraz utrata informacji zawartej w sekwencji DNA podczas przetwarzania danych. 
Zamiast regresji można było spróbować zastosować bardziej zaawansowane modele nieliniowe, takie jak Random Forest Regressor, Gradient Boosting Regressor lub MLPRegressor. Te modele prawdopodobnie lepiej, by odwzorowały złożone zależności biologiczne.
"""
