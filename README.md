# AmExpert 2021: Credit Card Default Prediction

Binary classification of credit card defaulters for American Express's AmExpert 2021 CodeLab on HackerEarth. The notebook uses PyCaret to benchmark many classifiers quickly, then tunes, blends and stacks the strongest ones against the competition's macro-F1 metric.

**Stack:** Python · PyCaret · scikit-learn · XGBoost · pandas

---

## Problem

Predict whether a customer will default on their credit card (`credit_card_default`) from demographic, income and credit-history features.

| Column | Description |
|---|---|
| `customer_id`, `name` | Identifiers (excluded from modeling) |
| `age`, `gender`, `no_of_children`, `total_family_members` | Demographics |
| `owns_car`, `owns_house`, `migrant_worker`, `occupation_type` | Assets and employment |
| `net_yearly_income`, `no_of_days_employed`, `yearly_debt_payments` | Income and obligations |
| `credit_limit`, `credit_limit_used(%)`, `credit_score` | Credit profile |
| `prev_defaults`, `default_in_last_6months` | Default history |
| **`credit_card_default`** | **Target** |

**Metric:** `100 * f1_score(actual, predicted, average='macro')`. Macro-F1 weights the default and non-default classes equally, which matters because defaulters are only about 8% of the 45,528 training rows (3,697 defaults): a model that predicts "no default" for everyone scores well on accuracy but poorly here.

## Approach

1. **Set up a PyCaret experiment** (`setup(data=df, target='credit_card_default')`), using its default preprocessing, hold-out split and stratified cross-validation.
2. **Compare models.** `compare_models()` ranks the available classifiers on cross-validated metrics in one call.
3. **Tune for the right metric.** Random Forest and XGBoost were tuned with `tune_model(..., optimize='F1')`, so hyperparameter search targets F1 rather than PyCaret's default (accuracy).
4. **Combine models.** A blend of selected models and a stacked ensemble were built and compared against the tuned single models.
5. **Finalize and predict.** The chosen model is refit on the full training data with `finalize_model()` and scored on the test set with `predict_model()`.

## Repository layout

```
AmExpert-2021-PyCaret.ipynb   experiment setup, comparison, tuning, blending, stacking
train.csv · test.csv          competition data
```

## Why PyCaret

PyCaret wraps scikit-learn, XGBoost, LightGBM, CatBoost and others behind a low-code API. In a time-boxed competition, it turned model selection from hundreds of lines of boilerplate into a few calls, leaving more time for choosing the right metric and ensembling strategy.
