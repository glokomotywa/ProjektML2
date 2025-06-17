## Jakub Woś 288581 Informatyka Ogólnoakademicka
## Projekt 4
Skrypt breast_cancer.py demonstruje prosty pipeline uczenia maszynowego w Pythonie na klasycznym zestawie danych Breast Cancer Wisconsin dostarczanym przez bibliotekę scikit-learn.
Celem jest porównanie dwóch popularnych klasyfikatorów:

   - Decision Tree (Drzewo decyzyjne)

   - k-Nearest Neighbours (k-NN)

Porównanie opiera się na dwóch kluczowych metrykach: Accuracy oraz ROC AUC.
### Wymagania
```bash
Python >= 3.9
numpy
pandas
matplotlib
scikit-learn
```
Instalacja wymagań:
```bash
pip install -r requirements.txt
```
### Instrukcja uruchomienia

```bash
python breast_cancer.py
```

Skrypt:

   - pobiera dane,

   - uczy oba modele na 75 % danych treningowych,

   - ocenia je na 25 % danych testowych,

   - drukuje tabelę metryk,

   - wyświetla krzywe ROC oraz wykres słupkowy z porównaniem metryk. 

### Struktura skryptu
| Sekcja	| Linia w kodzie	| Opis|
|--------|-----------------|-----|
|1. Wczytanie danych	| load_breast_cancer	| Ładuje cechy i etykiety z wbudowanego zbioru
2. Podział danych	|train_test_split	|75 % train / 25 % test, z podziałem stratyfikowanym
3. Drzewo decyzyjne	|DecisionTreeClassifier	|Trening, predykcja, proba klas
4. k-NN z normalizacją|	StandardScaler + KNeighborsClassifier|	Skalowanie cech i trening k-NN
5. Metryki	|accuracy_score, roc_auc_score	|Tworzy metrics_df ze zbiorczymi wynikami
6. Krzywe ROC	|RocCurveDisplay.from_predictions	|Dwie krzywe w jednym wykresie
7. Wykres słupkowy	|plt.bar	|Porównanie Accuracy i ROC AUC

### Opis metryk i wyników

Porównanie metryk:

| Model | Accuracy | ROC AUC |
|-------|----------|---------|
|  Drzewo decyzyjne | 0.92 |    0.92
|  k-NN | 0.98  |   0.9|

W powyższym uruchomieniu k-NN przewyższa drzewo decyzyjne w obu metrykach.
Wizualizacje

- Krzywe ROC – prezentują czułość (TPR) vs. 1-specyficzność (FPR) dla obu modeli na jednym wykresie.
- Wykres słupkowy – zestawia wartości Accuracy i ROC AUC obok siebie, ułatwiając szybkie porównanie.

Oba wykresy wyświetlane są na żywo dzięki matplotlib i nie zapisują się automatycznie do pliku. 
