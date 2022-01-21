![AmExpert-2021](https://i.imgur.com/r2WZwuh.png)

# AmExpert-2021-PyCaret 🏦

## Task 💼
Competetion conducted by American Express on [HackerEarth Platform](https://www.hackerearth.com/challenges/competitive/amexpert-code-lab/).
We are given relevant information about the customers of a company. And we're required to build a Machine Learning Model that can predict if there will be Credit Card Defaulters.

## Dataset 📚

| Column Name             | Description                                 |
| ----------------------- | ------------------------------------------- |
| customer_id             | unique identification of customer           |
| name                    | name of customer                            |
| age                     | age of customer (Years)                     |
| gender                  | gender of customer (M or F)                 |
| owns_car                | whether a customer owns a car (Y or N)      |
| owns_house              | whether a customer owns a house (Y or N)    |
| no_of_children          | number of children of a customer            |
| net_yearly_income       | net yearly income of a customer (USD)       |
| no_of_days_employed     | no. of days employed                        |
| occupation_type         | occupation type of customer                 |
| total_family_members    | no. of family members of customer           |
| migrant_worker          | customer is migrant worker (Yes or No)      |
| yearly_debt_payments    | yearly debt of customer (USD)               |
| credit_limit            | credit limit of customer (USD)              |
| credit_limit_used(%)    | credit limit used by customer               |
| credit_score            | credit score of customer                    |       
| prev_defaults           | no. of previous defaults                    |
| default_in_last_6months | whether a customer has defaulted (Yes or No)|
| **credit_card_default** | whether there will be credit card default (Yes or No) |

## Evaluation Metrics 💯
**Score:** 100 * (metrics.f1_score(actual, predicted, average='macro'))

## ML Library: PyCaret 🔥
* PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

* In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and few more.

* The design and simplicity of PyCaret are inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more technical expertise.
