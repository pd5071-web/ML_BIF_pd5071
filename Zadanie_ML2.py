# pd5071
# Uczenie maszynowe w bioinformatyce
# Zadanie ML2

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
    'KRAS': [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['TP53_expr',
        'BRCA1_expr',
        'TF_motifs',
        'KRAS']]

y = df['Cancer_status']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Komentarz:
# Model drzewa decyzyjnego oparty na danych ekspresji genów TP53, BRCA1, TF_motifs i KRAS.
# Accuracy określa procent poprawnych klasyfikacji.
# Precision określa skuteczność wykrywania nowotworów.
# Recall pokazuje jaki odsetek rzeczywistych przypadków nowotworu został wykryty.
# F1-score stanowi kompromis pomiędzy precision i recall.
# Wyniki: Accuracy: 1.0; Precision: 1.0; Recall: 1.0; F1-score: 1.0
# Wynik ten jest prawdopodobnie związany z bardzo dobrą separacją klas w danych oraz niewielką liczbą obserwacji (8 rekordów).
# Tak mały zbiór danych przy  pojedynczym podziale train/test może prowadzić do zawyżonej oceny jakości modelu.
