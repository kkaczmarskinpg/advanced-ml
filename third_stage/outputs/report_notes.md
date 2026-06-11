# Etap 3 - notatki do raportu

## Punkt odniesienia
Baseline XGBoost z etapu 2 uzyskal na zbiorze testowym F1 = 0.8149, recall = 0.7625, precision = 0.8749.

## Optymalizacja cech
W macierzy korelacji znaleziono 5 par cech o |r| >= 0.8.
Najlepszy wariant SelectKBest w CV wybral k = 22 z F1 CV = 0.8070 +/- 0.0054.

## Najlepszy wariant
Najwyzsze F1 na zbiorze testowym ma `xgboost_genetic`: F1 = 0.8201, recall = 0.7715, precision = 0.8752, ROC-AUC = 0.9642, PR-AUC = 0.9113.

## Artefakty
- `stage3_results.csv` - tabela zbiorcza metryk.
- `correlation_heatmap.png` i `correlation_matrix.csv` - analiza korelacji.
- `select_k_best_cv_results.csv` - wyniki selekcji cech.
- `*_feature_importance.csv/png` - waznosc cech.
- `*_local_xgb_contributions.csv` - lokalne wyjasnienia predykcji XGBoost.
