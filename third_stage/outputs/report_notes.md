# Etap 3 - notatki do raportu

## 1. Cel etapu

Celem trzeciego etapu była optymalizacja najlepszego modelu z etapu 2, czyli XGBoost, oraz sprawdzenie, czy dalsze strojenie cech i hiperparametrów poprawia jakość klasyfikacji. Najważniejszą metryką porównawczą było F1, ponieważ zbiór jest niezbalansowany. Dodatkowo analizowano recall klasy pozytywnej, precision, specificity, ROC-AUC, PR-AUC, macierz pomyłek, czas treningu oraz liczbę wykorzystanych cech.

Klasa pozytywna to `loan_status = 1`. Wszystkie warianty XGBoost były oceniane przy progu decyzyjnym `0.5`; FLAML był oceniany przez własną metodę `predict`.

## 2. Dane i walidacja

| Zbiór | Liczba obserwacji | Udział klasy pozytywnej |
|---|---:|---:|
| train | 28 800 | 22.22% |
| validation | 7 200 | 22.22% |
| train + validation | 36 000 | 22.22% |
| test | 9 000 | 22.22% |

Do treningu i strojenia wykorzystano połączony zbiór `train + validation`, a końcowe porównanie wykonano na niezależnym zbiorze testowym. W eksperymentach optymalizacyjnych zastosowano `StratifiedKFold` z 5 foldami, tasowaniem i stałym `random_state = 42`. Główną metryką w walidacji krzyżowej było F1.

## 3. Punkt odniesienia: XGBoost baseline

Baseline odpowiada modelowi XGBoost z etapu 2 uruchomionemu na tych samych danych i metrykach. Parametry bazowe:

| Parametr | Wartość |
|---|---:|
| `n_estimators` | 400 |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `eval_metric` | logloss |

Wynik baseline na zbiorze testowym:

| Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9230 | 0.8749 | 0.7625 | 0.8149 | 0.9689 | 0.9639 | 0.9106 | 6782 | 218 | 475 | 1525 |

Baseline jest mocnym punktem odniesienia: ma wysoką precision i specificity, czyli popełnia relatywnie mało fałszywych alarmów. Główne pole do poprawy to recall klasy pozytywnej, ponieważ 475 przypadków pozytywnych zostało sklasyfikowanych jako negatywne.

## 4. Optymalizacja cech

### 4.1. Macierz korelacji

W macierzy korelacji Pearsona znaleziono 5 par cech o `|r| >= 0.8`.

| Cecha 1 | Cecha 2 | Korelacja |
|---|---|---:|
| `male` | `female` | -1.0000 |
| `person_age` | `person_emp_exp` | 0.9601 |
| `rent` | `mortage` | -0.8818 |
| `person_age` | `cb_person_cred_hist_length` | 0.8740 |
| `person_emp_exp` | `cb_person_cred_hist_length` | 0.8377 |

Najsilniejsza zależność dotyczy pary `male` / `female`, co wynika z kodowania binarnego tej samej informacji. Wysoką korelację mają też cechy związane z wiekiem, doświadczeniem zawodowym i długością historii kredytowej. Po usunięciu cech redundantnych model `xgboost_correlation_pruned` używał 18 cech zamiast 22, ale jego F1 spadło z 0.8149 do 0.8127, a recall z 0.7625 do 0.7580. Oznacza to, że redukcja cech uprościła model, ale nie poprawiła jakości predykcji.

### 4.2. SelectKBest

SelectKBest z funkcją `f_classif` sprawdzono dla kilku wartości `k`.

| k | Średnie F1 CV | Odchylenie std |
|---:|---:|---:|
| 22 | 0.8070 | 0.0054 |
| 16 | 0.8043 | 0.0057 |
| 18 | 0.8026 | 0.0056 |
| 14 | 0.8022 | 0.0039 |
| 20 | 0.8020 | 0.0049 |
| 12 | 0.7997 | 0.0049 |
| 10 | 0.7911 | 0.0081 |
| 8 | 0.7876 | 0.0062 |

Najlepszy wynik CV uzyskało `k = 22`, czyli pełny zestaw cech. Wariant `xgboost_select_k_best` ma na teście identyczny wynik jak baseline: F1 = 0.8149. Wniosek jest taki, że SelectKBest nie wskazał bezpiecznej redukcji wymiarowości; usuwanie cech pogarszało średni wynik CV.

Najwyżej ocenione cechy według SelectKBest:

| Cecha | Score |
|---|---:|
| `previous_loan_defaults_on_file` | 14077.92 |
| `loan_percent_income` | 6104.45 |
| `loan_int_rate` | 4403.72 |
| `rent` | 2521.51 |
| `mortage` | 1818.86 |
| `person_income` | 583.82 |
| `loan_amnt` | 413.35 |
| `venture` | 298.78 |
| `own` | 275.47 |
| `debtconsolidation` | 262.56 |

### 4.3. Ważność cech XGBoost

Najważniejsze cechy w najlepszym modelu XGBoost po optymalizacji genetycznej:

| Cecha | Importance |
|---|---:|
| `previous_loan_defaults_on_file` | 0.7686 |
| `rent` | 0.0345 |
| `own` | 0.0318 |
| `mortage` | 0.0216 |
| `loan_percent_income` | 0.0178 |
| `venture` | 0.0160 |
| `loan_int_rate` | 0.0140 |
| `person_income` | 0.0126 |
| `homeimprovement` | 0.0118 |
| `debtconsolidation` | 0.0093 |

Najsilniejszą cechą jest `previous_loan_defaults_on_file`. Jej dominacja pojawia się zarówno w SelectKBest, jak i w ważności XGBoost, więc jest to najbardziej stabilny sygnał predykcyjny. Istotne są też relacja kwoty raty/pożyczki do dochodu, oprocentowanie, status mieszkaniowy oraz cel pożyczki. Wariant oparty na cechach z ważności (`xgboost_top_importance_features`) nie poprawił wyniku i osiągnął F1 = 0.8114, więc nie został wybrany.

## 5. Optymalizacja hiperparametrów

### 5.1. Grid Search

Grid Search sprawdzał kombinacje `n_estimators`, `max_depth`, `learning_rate` i `scale_pos_weight`. Najlepsza konfiguracja w CV:

| Parametr | Wartość |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 1.0 |
| Średnie F1 CV | 0.8072 |
| Std F1 CV | 0.0062 |

Na zbiorze testowym Grid Search poprawił F1 względem baseline z 0.8149 do 0.8157. Poprawa jest bardzo mała, ale stabilna: precision wzrosło z 0.8749 do 0.8756, recall z 0.7625 do 0.7635, a PR-AUC z 0.9106 do 0.9106 po zaokrągleniu.

### 5.2. Optuna

Optuna wykonała 50 prób i optymalizowała średnie F1 z walidacji krzyżowej. Najlepszy trial osiągnął F1 CV = 0.8107.

| Parametr | Wartość |
|---|---:|
| `n_estimators` | 600 |
| `max_depth` | 6 |
| `learning_rate` | 0.0534 |
| `subsample` | 0.8024 |
| `colsample_bytree` | 0.9446 |
| `min_child_weight` | 2 |
| `gamma` | 1.8664 |
| `reg_alpha` | 0.0004 |
| `reg_lambda` | 2.8110 |
| `scale_pos_weight` | 1.2485 |

Na zbiorze testowym Optuna uzyskała F1 = 0.8171. Jest to wynik lepszy od baseline i Grid Search. Optuna miała też najwyższy recall ze wszystkich wariantów: 0.7840, czyli wykryła 1568 przypadków pozytywnych wobec 1525 w baseline. Odbyło się to kosztem precision, które spadło z 0.8749 do 0.8531, oraz większej liczby false positive: 270 zamiast 218.

### 5.3. Algorytm genetyczny

Algorytm genetyczny przeszukiwał przestrzeń `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` i `min_child_weight`. Użyto populacji 20 osobników, 15 generacji, prawdopodobieństwa krzyżowania 0.8 i mutacji 0.1.

Najlepsza konfiguracja:

| Parametr | Wartość |
|---|---:|
| `n_estimators` | 562 |
| `max_depth` | 7 |
| `learning_rate` | 0.0611 |
| `subsample` | 0.9099 |
| `colsample_bytree` | 0.7381 |
| `min_child_weight` | 1 |
| Średnie F1 CV | 0.8096 |
| Std F1 CV | 0.0021 |

Na zbiorze testowym wariant genetyczny osiągnął najlepsze F1 = 0.8201. Poprawa względem baseline wynosi +0.0052 F1. Model zwiększył recall z 0.7625 do 0.7715 i utrzymał precision praktycznie na tym samym poziomie: 0.8752 wobec 0.8749. To najlepszy kompromis między wykrywaniem klasy pozytywnej a kontrolą liczby fałszywych alarmów.

Warto zauważyć, że Optuna miała wyższe F1 w CV niż algorytm genetyczny, ale na zbiorze testowym lepszy okazał się model genetyczny. Różnice są małe, więc należy traktować je jako umiarkowaną, a nie radykalną poprawę.

## 6. AutoML

W ramach AutoML użyto FLAML z budżetem czasu 180 sekund i metryką `f1`. Przeszukiwane były estymatory: XGBoost, LightGBM, Random Forest, Extra Trees, CatBoost i regresja logistyczna L1. Najlepszym estymatorem FLAML był `lgbm`.

Najlepsza konfiguracja FLAML:

| Parametr | Wartość |
|---|---:|
| `best_estimator` | `lgbm` |
| `n_estimators` | 742 |
| `num_leaves` | 11 |
| `learning_rate` | 0.1138 |
| `colsample_bytree` | 0.8100 |
| `min_child_samples` | 15 |
| `reg_alpha` | 0.3702 |
| `reg_lambda` | 0.0497 |
| `log_max_bin` | 10 |

FLAML osiągnął F1 = 0.8165, czyli wynik lepszy od baseline i Grid Search, ale słabszy niż Optuna i algorytm genetyczny. AutoML był więc konkurencyjny, ale nie znalazł najlepszego rozwiązania w tym etapie.

## 7. Porównanie końcowe

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Cechy | Czas [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xgboost_genetic` | 0.9248 | 0.8752 | 0.7715 | 0.8201 | 0.9686 | 0.9642 | 0.9113 | 22 | 582.85 |
| `xgboost_optuna` | 0.9220 | 0.8531 | 0.7840 | 0.8171 | 0.9614 | 0.9637 | 0.9099 | 22 | 407.05 |
| `flaml_automl` | 0.9234 | 0.8735 | 0.7665 | 0.8165 | 0.9683 | 0.9638 | 0.9109 | 22 | 180.56 |
| `xgboost_grid_search` | 0.9233 | 0.8756 | 0.7635 | 0.8157 | 0.9690 | 0.9640 | 0.9106 | 22 | 22.37 |
| `xgboost_select_k_best` | 0.9230 | 0.8749 | 0.7625 | 0.8149 | 0.9689 | 0.9639 | 0.9106 | 22 | 0.47 |
| `xgboost_baseline` | 0.9230 | 0.8749 | 0.7625 | 0.8149 | 0.9689 | 0.9639 | 0.9106 | 22 | 1.01 |
| `xgboost_correlation_pruned` | 0.9223 | 0.8758 | 0.7580 | 0.8127 | 0.9693 | 0.9634 | 0.9095 | 18 | 0.69 |
| `xgboost_top_importance_features` | 0.9219 | 0.8755 | 0.7560 | 0.8114 | 0.9693 | 0.9632 | 0.9096 | 22 | 0.49 |

Zmiany względem baseline:

| Model | ΔF1 | ΔPrecision | ΔRecall | ΔROC-AUC | ΔPR-AUC |
|---|---:|---:|---:|---:|---:|
| `xgboost_genetic` | +0.0052 | +0.0003 | +0.0090 | +0.0003 | +0.0007 |
| `xgboost_optuna` | +0.0022 | -0.0218 | +0.0215 | -0.0002 | -0.0007 |
| `flaml_automl` | +0.0017 | -0.0014 | +0.0040 | -0.0001 | +0.0003 |
| `xgboost_grid_search` | +0.0009 | +0.0006 | +0.0010 | +0.0001 | +0.0001 |
| `xgboost_select_k_best` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `xgboost_correlation_pruned` | -0.0022 | +0.0009 | -0.0045 | -0.0005 | -0.0011 |
| `xgboost_top_importance_features` | -0.0035 | +0.0006 | -0.0065 | -0.0007 | -0.0010 |

Macierze pomyłek na zbiorze testowym:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `xgboost_genetic` | 6780 | 220 | 457 | 1543 |
| `xgboost_optuna` | 6730 | 270 | 432 | 1568 |
| `flaml_automl` | 6778 | 222 | 467 | 1533 |
| `xgboost_grid_search` | 6783 | 217 | 473 | 1527 |
| `xgboost_baseline` | 6782 | 218 | 475 | 1525 |
| `xgboost_correlation_pruned` | 6785 | 215 | 484 | 1516 |
| `xgboost_top_importance_features` | 6785 | 215 | 488 | 1512 |

Najlepszym wariantem według F1 jest `xgboost_genetic`. Jeżeli priorytetem byłoby maksymalne wykrywanie klasy pozytywnej, można rozważyć `xgboost_optuna`, bo ma najwyższy recall. W głównym kryterium projektu, czyli F1 przy zachowaniu dobrego precision, wariant genetyczny jest jednak bardziej zrównoważony.

## 8. Wyjaśnialność najlepszego modelu

Dla najlepszego wariantu zapisano globalne i lokalne artefakty interpretacyjne:

- `xgboost_genetic_feature_importance.csv` i `.png` - ważność cech XGBoost,
- `xgboost_genetic_shap_summary.png` - rozkład wpływu cech SHAP,
- `xgboost_genetic_shap_bar.png` - średnia ważność bezwzględna SHAP,
- `xgboost_genetic_local_xgb_contributions.csv` - lokalne kontrybucje dla przykładów TP, TN, FP i FN.

Globalnie najważniejsza jest cecha `previous_loan_defaults_on_file`, a dalej status mieszkaniowy, `loan_percent_income`, `loan_int_rate`, cel pożyczki i dochód. Model opiera się więc na zmiennych zgodnych z intuicją domenową: historii wcześniejszych niespłat, obciążeniu pożyczką względem dochodu, koszcie kredytu i kontekście finansowym klienta.

Lokalne przykłady:

| Przypadek | Prawdziwa klasa | Predykcja | Prawdopodobieństwo klasy 1 | Najsilniejsze dodatnie kontrybucje | Najsilniejsze ujemne kontrybucje |
|---|---:|---:|---:|---|---|
| true positive | 1 | 1 | 0.8119 | `loan_int_rate` (+2.4188), `previous_loan_defaults_on_file` (+1.3718), `debtconsolidation` (+0.9292) | `person_income` (-0.9607), `person_emp_exp` (-0.3086), `rent` (-0.2738) |
| true negative | 0 | 0 | 0.0024 | `loan_amnt` (+0.1607), `credit_score` (+0.1547), `person_age` (+0.1437) | `previous_loan_defaults_on_file` (-2.4149), `loan_int_rate` (-1.3598), `venture` (-0.8724) |
| false positive | 0 | 1 | 0.5508 | `previous_loan_defaults_on_file` (+1.1500), `person_income` (+0.8499), `credit_score` (+0.2004) | `loan_int_rate` (-0.5817), `loan_percent_income` (-0.3192), `homeimprovement` (-0.1387) |
| false negative | 1 | 0 | 0.4958 | `previous_loan_defaults_on_file` (+1.2415), `credit_score` (+0.5272), `debtconsolidation` (+0.3203) | `loan_percent_income` (-0.4858), `loan_int_rate` (-0.3371), `person_income` (-0.2064) |

Przypadki błędne są blisko progu decyzyjnego. False positive ma prawdopodobieństwo 0.5508, a false negative 0.4958, więc niewielka zmiana progu mogłaby zmienić decyzję. To sugeruje, że w dalszej pracy warto rozważyć strojenie thresholda zależnie od kosztu FP i FN.

## 9. Wnioski końcowe

1. Optymalizacja hiperparametrów poprawiła wynik względem baseline, ale skala poprawy jest umiarkowana. Najlepszy model `xgboost_genetic` poprawił F1 z 0.8149 do 0.8201.
2. Selekcja cech nie poprawiła modelu. SelectKBest wybrał pełne `k = 22`, a usuwanie cech skorelowanych obniżyło F1 do 0.8127.
3. Grid Search dał minimalną poprawę względem baseline: F1 = 0.8157.
4. Optuna dała lepszy wynik niż Grid Search: F1 = 0.8171 oraz najwyższy recall = 0.7840, ale kosztem niższego precision.
5. Algorytm genetyczny dał najlepszy wynik testowy: F1 = 0.8201, recall = 0.7715 i precision = 0.8752.
6. AutoML FLAML był konkurencyjny, ale nie lepszy od najlepszego ręcznie optymalizowanego XGBoost: F1 = 0.8165.
7. Najważniejsze cechy są interpretowalne i domenowo sensowne: wcześniejsze niespłaty, relacja pożyczki do dochodu, oprocentowanie, status mieszkaniowy, dochód i cel pożyczki.
8. Rekomendowany model końcowy to `xgboost_genetic`, ponieważ ma najwyższe F1 i poprawia recall bez zauważalnego spadku precision.

## 10. Artefakty

| Plik | Zawartość |
|---|---|
| `stage3_results.csv` | Pełna tabela metryk, macierzy pomyłek, czasów treningu i parametrów |
| `final_comparison_table.csv` | Skrócona tabela porównawcza do raportu |
| `split_summary.csv` | Liczności i udział klasy pozytywnej w podziałach danych |
| `correlation_matrix.csv` | Macierz korelacji Pearsona |
| `correlation_heatmap.png` | Wizualizacja macierzy korelacji |
| `high_correlation_pairs.csv` | Pary cech z `|r| >= 0.8` |
| `select_k_best_cv_results.csv` | Wyniki CV dla różnych wartości `k` |
| `select_k_best_feature_scores.csv` | Score i p-value cech w SelectKBest |
| `grid_search_results.csv` | Szczegółowe wyniki Grid Search |
| `optuna_trials.csv` | Historia prób Optuny |
| `optuna_best_params.json` | Najlepsze parametry Optuny |
| `genetic_search_results.csv` | Szczegółowe wyniki algorytmu genetycznego |
| `flaml_best_config.json` | Najlepsza konfiguracja FLAML |
| `xgboost_baseline_feature_importance.csv/png` | Ważność cech baseline |
| `xgboost_genetic_feature_importance.csv/png` | Ważność cech najlepszego XGBoost |
| `xgboost_genetic_shap_summary.png` | SHAP summary plot |
| `xgboost_genetic_shap_bar.png` | SHAP bar plot |
| `xgboost_genetic_local_xgb_contributions.csv` | Lokalne wyjaśnienia predykcji najlepszego modelu |
