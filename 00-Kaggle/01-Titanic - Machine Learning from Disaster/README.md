\# Titanic: Machine Learning from Disaster



A structured machine-learning experiment using Kaggle’s Titanic classification competition.



The goal was not only to improve leaderboard accuracy, but to build a reproducible experimentation workflow, compare model families, test validation assumptions, and document why candidate models were accepted or rejected.



\## Final Result



| Item                      | Result                |

| ------------------------- | --------------------- |

| Task                      | Binary classification |

| Metric                    | Accuracy              |

| Champion model            | Random Forest         |

| Cross-validation accuracy | 0.8316                |

| Kaggle accuracy           | \*\*0.78708\*\*           |

| Experiments completed     | 10                    |



\## Champion Model



The strongest submission used a regularized Random Forest:



```python

RandomForestClassifier(

&#x20;   n\_estimators=500,

&#x20;   max\_depth=5,

&#x20;   min\_samples\_split=8,

&#x20;   min\_samples\_leaf=4,

&#x20;   max\_features="sqrt",

&#x20;   random\_state=42,

&#x20;   n\_jobs=-1,

)

```



\## Features



The model used the original passenger variables alongside several engineered features:



\* `Title`

\* `FamilySize`

\* `IsAlone`

\* `Deck`

\* `FarePerPerson`



Categorical features were one-hot encoded. Missing numerical values were imputed with the median, while missing categorical values were imputed with the most frequent category.



\## Validation Strategy



Five-fold stratified cross-validation was used as the primary validation method.



A secondary family-group-aware validation scheme was also tested to determine whether related passengers appearing across folds caused meaningful leakage. Group-aware accuracy was slightly lower and substantially more variable, but did not indicate enough leakage to replace ordinary stratified validation.



\## Experiment Summary



| Experiment | Change                          |     CV |      Kaggle | Decision         |

| ---------- | ------------------------------- | -----: | ----------: | ---------------- |

| EXP-001    | XGBoost baseline                | 0.8372 |     0.76076 | Rejected         |

| EXP-002    | Logistic regression             | 0.8283 |     0.76555 | Rejected         |

| EXP-003    | Regularized Random Forest       | 0.8316 | \*\*0.78708\*\* | \*\*Champion\*\*     |

| EXP-004    | Logistic interaction features   | 0.8339 |     0.76315 | Rejected         |

| EXP-005    | Group-based age imputation      | 0.8294 |     0.78229 | Rejected         |

| EXP-006    | Family-group validation         | 0.8270 |           — | Validation study |

| EXP-007    | Random Forest tuning            |      — |           — | Rejected locally |

| EXP-008    | Ticket and prefix features      | 0.8224 |           — | Rejected locally |

| EXP-009    | Ticket-group features           | 0.8301 |           — | Rejected locally |

| EXP-010    | Random Forest/logistic ensemble | 0.8330 |     0.77751 | Rejected         |



Detailed experiment notes are available in \[`experiments.csv`](experiments.csv).



\## Key Findings



\### Higher validation accuracy did not always transfer



XGBoost produced the strongest initial cross-validation score but the weakest Kaggle result among the first three model families.



The Random Forest achieved slightly lower local validation accuracy but generalized substantially better to the hidden test set.



\### Small prediction changes created large uncertainty



Several candidate models changed only a handful of the 418 test predictions. Even candidates that improved multiple local validation schemes sometimes performed worse on Kaggle.



\### Additional complexity was not automatically useful



Interaction features, grouped age imputation, ticket features, hyperparameter tuning, and probability ensembling did not reliably improve the Random Forest champion.



\## Repository Contents



```text

.

├── README.md

├── titanic\_experiments.ipynb

├── experiments.csv

├── requirements.txt

├── data/

├── results/

└── submissions/

```



\## Running the Project



1\. Download the Titanic competition data from Kaggle.

2\. Place `train.csv` and `test.csv` inside the local `data/` folder.

3\. Install the dependencies:



```bash

pip install -r requirements.txt

```



4\. Run `titanic\_experiments.ipynb` from top to bottom.



\## Final Takeaway



The best result came from a constrained Random Forest rather than the model with the highest initial cross-validation score.



The broader lesson was the importance of controlled experiments, validation robustness, documenting failed ideas, and stopping once further work becomes leaderboard chasing rather than transferable learning.



