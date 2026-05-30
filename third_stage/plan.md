# Plan realizacji etapu 3 projektu

## 1. Cel etapu 3

Celem trzeciego etapu projektu będzie optymalizacja najlepszego modelu z etapu 2, czyli modelu **XGBoost**, oraz dokładne porównanie wyników przed i po optymalizacji. W etapie 2 XGBoost osiągnął najlepszy wynik F1 na zbiorze testowym, dlatego zgodnie z wymaganiami dalsze strojenie zostanie wykonane tylko dla tego modelu.

Główne zadania etapu 3 będą obejmować:

* optymalizację cech,
* optymalizację hiperparametrów z użyciem walidacji krzyżowej,
* porównanie wyników z modelem baseline,
* implementację wybranego rozwiązania AutoML,
* analizę wpływu optymalizacji na wyniki,
* dodanie wyjaśnialności modelu.

## 2. Punkt odniesienia: model baseline

Jako model baseline zostanie przyjęty XGBoost z etapu 2, wytrenowany na tych samych danych i oceniony tymi samymi metrykami. Wyniki baseline będą stanowiły punkt odniesienia dla wszystkich kolejnych eksperymentów.

Dla baseline zostaną zapisane następujące metryki:

* accuracy,
* precision,
* recall,
* F1,
* specificity,
* ROC-AUC,
* PR-AUC,
* macierz pomyłek.

Najważniejszą metryką porównawczą będzie **F1**, ponieważ problem klasyfikacyjny jest niezbalansowany. Dodatkowo szczególna uwaga zostanie zwrócona na recall dla klasy pozytywnej, ponieważ zależy nam na skutecznym wykrywaniu klasy mniejszościowej.

## 3. Optymalizacja cech

W pierwszej części etapu 3 zostanie przeprowadzona analiza i selekcja cech. Celem będzie sprawdzenie, czy usunięcie cech słabych, redundantnych lub silnie skorelowanych pozwala poprawić jakość modelu albo uprościć go bez pogorszenia wyników.

### 3.1. Macierz korelacji

Pierwszą obowiązkową metodą będzie przygotowanie macierzy korelacji dla cech numerycznych. Na jej podstawie zostaną wskazane pary cech silnie ze sobą skorelowanych.

Planowane działania:

* obliczenie korelacji pomiędzy cechami,
* wizualizacja macierzy korelacji w formie heatmapy,
* wskazanie par cech o wysokiej korelacji,
* analiza, czy część cech można usunąć jako redundantne,
* porównanie wyników modelu przed i po ewentualnym usunięciu cech.

### 3.2. SelectKBest

Drugą metodą optymalizacji cech będzie **SelectKBest**. Metoda ta pozwoli wybrać określoną liczbę najbardziej informacyjnych cech względem zmiennej docelowej.

Planowane działania:

* zastosowanie SelectKBest dla różnych wartości `k`,
* wykorzystanie odpowiedniej funkcji oceny, np. `f_classif` lub `mutual_info_classif`,
* trening XGBoost na wybranych podzbiorach cech,
* ocena wyników z użyciem walidacji krzyżowej,
* wybór liczby cech dającej najlepszy kompromis między jakością a prostotą modelu.

### 3.3. Dodatkowa analiza ważności cech

Dodatkowo zostanie wykorzystana ważność cech z modelu XGBoost. Pozwoli to porównać wyniki metod statystycznych z tym, jak cechy są wykorzystywane przez sam model.

Planowane działania:

* wyznaczenie feature importance dla modelu XGBoost,
* porównanie najważniejszych cech z wynikami SelectKBest,
* sprawdzenie, czy ograniczenie modelu do najważniejszych cech poprawia lub utrzymuje wynik F1,
* przygotowanie wykresu najważniejszych cech.

## 4. Optymalizacja hiperparametrów modelu XGBoost

Optymalizacja hiperparametrów zostanie wykonana z użyciem walidacji krzyżowej. Ponieważ wymagania zakładają, że wyższa ocena powinna zawierać elementy z ocen niższych, plan obejmie zarówno **Grid Search**, jak i **Optunę**. Dodatkowo można rozszerzyć projekt o algorytm genetyczny, jeżeli celem będzie realizacja pełnego wariantu na najwyższą ocenę.

### 4.1. Grid Search

Pierwszą metodą strojenia hiperparametrów będzie Grid Search. Zostanie przygotowana siatka najważniejszych parametrów XGBoost.

Przykładowe hiperparametry do optymalizacji:

* `n_estimators`,
* `max_depth`,
* `learning_rate`,
* `subsample`,
* `colsample_bytree`,
* `min_child_weight`,
* `gamma`,
* `reg_alpha`,
* `reg_lambda`,
* `scale_pos_weight`.

Każda kombinacja parametrów zostanie oceniona przy użyciu walidacji krzyżowej, np. StratifiedKFold z 5 foldami. Główną metryką wyboru najlepszego zestawu będzie F1.

Wyniki Grid Search zostaną zestawione z modelem baseline.

### 4.2. Optuna

Drugą metodą optymalizacji hiperparametrów będzie Optuna. W przeciwieństwie do Grid Search, Optuna nie sprawdza sztywnej siatki parametrów, lecz inteligentnie przeszukuje przestrzeń hiperparametrów.

Planowane działania:

* zdefiniowanie funkcji celu maksymalizującej średnie F1 z walidacji krzyżowej,
* określenie zakresów dla hiperparametrów XGBoost,
* uruchomienie optymalizacji dla ustalonej liczby prób, np. 50–100 triali,
* zapisanie najlepszego zestawu hiperparametrów,
* porównanie wyniku Optuny z wynikiem Grid Search i baseline.

Optuna powinna pozwolić znaleźć lepszy lub bardziej optymalny zestaw parametrów niż klasyczny Grid Search, szczególnie przy większej liczbie hiperparametrów.

### 4.3. Opcjonalnie: algorytm genetyczny

Jeżeli projekt będzie rozszerzany do wariantu na najwyższą ocenę, można dodatkowo zastosować algorytm genetyczny do optymalizacji hiperparametrów.

Planowane działania:

* zakodowanie hiperparametrów jako osobników populacji,
* ocena osobników na podstawie F1 z walidacji krzyżowej,
* zastosowanie selekcji, krzyżowania i mutacji,
* wybór najlepszego zestawu hiperparametrów,
* porównanie wyniku z Grid Search i Optuną.

Ta część pozwoli sprawdzić, czy metoda ewolucyjna daje dodatkową poprawę względem klasycznych i bayesowskich metod optymalizacji.

## 5. Walidacja krzyżowa

Wszystkie eksperymenty optymalizacyjne będą wykonywane z użyciem walidacji krzyżowej. Zostanie zastosowany StratifiedKFold, aby zachować proporcje klas w każdym foldzie.

Planowana konfiguracja:

* `StratifiedKFold`,
* 5 foldów,
* tasowanie danych,
* stałe `random_state`,
* główna metryka: F1,
* metryki dodatkowe: accuracy, precision, recall, specificity, ROC-AUC i PR-AUC.

Dzięki temu wyniki będą bardziej stabilne i mniej zależne od jednego losowego podziału danych.

## 6. AutoML

W etapie 3 zostanie zaimplementowany minimum jeden model AutoML. Proponowanym wyborem jest **FLAML**, **AutoGluon** albo **TPOT**.

Najbardziej praktycznym wyborem może być **FLAML**, ponieważ jest stosunkowo szybki i dobrze nadaje się do porównania z ręcznie strojonym modelem.

Planowane działania:

* przygotowanie danych w tym samym formacie jak dla modelu XGBoost,
* uruchomienie AutoML z ograniczeniem czasu treningu,
* wybór metryki optymalizacyjnej F1,
* zapisanie najlepszego modelu znalezionego przez AutoML,
* porównanie wyników AutoML z:

  * baseline XGBoost,
  * XGBoost po Grid Search,
  * XGBoost po Optunie,
  * opcjonalnie XGBoost po algorytmie genetycznym.

Celem AutoML będzie sprawdzenie, czy automatyczne wyszukiwanie modeli i hiperparametrów daje lepsze wyniki niż ręczna optymalizacja najlepszego modelu z etapu 2.

## 7. Porównanie wyników przed i po optymalizacji

Po zakończeniu eksperymentów zostanie przygotowana tabela zbiorcza z wynikami wszystkich wariantów.

Porównane zostaną następujące modele i konfiguracje:

1. XGBoost baseline z etapu 2,
2. XGBoost po selekcji cech,
3. XGBoost po Grid Search,
4. XGBoost po Optunie,
5. opcjonalnie XGBoost po algorytmie genetycznym,
6. najlepszy model AutoML.

Dla każdego wariantu zostaną podane:

* accuracy,
* precision,
* recall,
* F1,
* specificity,
* ROC-AUC,
* PR-AUC,
* czas treningu,
* liczba wykorzystanych cech,
* najlepsze hiperparametry.

W podsumowaniu zostanie wskazane:

* czy optymalizacja poprawiła wynik względem baseline,
* która metoda dała największy wzrost F1,
* czy poprawa F1 nie odbyła się kosztem dużego spadku recall albo precision,
* czy selekcja cech uprościła model bez pogorszenia wyników,
* czy AutoML okazał się lepszy od ręcznie optymalizowanego XGBoost.

## 8. Wyjaśnialność modelu

Ostatnim elementem etapu 3 będzie interpretacja najlepszego uzyskanego modelu. Do wyjaśnialności zostaną wykorzystane metody globalne i lokalne.

### 8.1. Globalna interpretacja modelu

Zostaną przygotowane:

* wykres ważności cech XGBoost,
* analiza najważniejszych zmiennych,
* porównanie ważności cech przed i po optymalizacji,
* opcjonalnie wykresy SHAP summary plot.

Ta część pokaże, które cechy miały największy wpływ na decyzje modelu.

### 8.2. Lokalna interpretacja predykcji

Dla kilku przykładowych obserwacji zostanie wykonana lokalna analiza predykcji, np. z użyciem SHAP.

Planowane działania:

* wybór kilku poprawnie i błędnie sklasyfikowanych przypadków,
* sprawdzenie, które cechy wpłynęły na decyzję modelu,
* porównanie przypadków false positive i false negative,
* opisanie, dlaczego model mógł podjąć daną decyzję.

Dzięki temu będzie można nie tylko ocenić jakość modelu liczbowo, ale także lepiej zrozumieć sposób jego działania.

## 9. Proponowana struktura raportu z etapu 3

Raport z etapu 3 może mieć następującą strukturę:

1. Wprowadzenie i cel etapu 3.
2. Przypomnienie wyników z etapu 2 i wybór modelu XGBoost.
3. Opis danych i metryk.
4. Wyniki modelu baseline.
5. Optymalizacja cech:

   * macierz korelacji,
   * SelectKBest,
   * ważność cech.
6. Optymalizacja hiperparametrów:

   * Grid Search,
   * Optuna,
   * opcjonalnie algorytm genetyczny.
7. Implementacja AutoML.
8. Porównanie wszystkich wyników.
9. Wyjaśnialność najlepszego modelu.
10. Wnioski końcowe.

## 10. Oczekiwany efekt końcowy

Efektem końcowym etapu 3 będzie wybór najlepszego wariantu modelu oraz dokładne uzasadnienie, czy optymalizacja rzeczywiście poprawiła jakość klasyfikacji. Najważniejsze będzie porównanie wyników względem baseline XGBoost z etapu 2.

Końcowe wnioski powinny odpowiedzieć na pytania:

* czy selekcja cech poprawiła lub uprościła model,
* czy Grid Search poprawił wynik baseline,
* czy Optuna dała lepszy wynik niż Grid Search,
* czy AutoML znalazł model lepszy od ręcznie optymalizowanego XGBoost,
* które cechy najbardziej wpływają na predykcję,
* czy najlepszy model jest nie tylko skuteczny, ale też możliwy do interpretacji.
