# Resume Bias Project

## About This Project

This project is deeply personal. As a Black woman named **Waithira**, I’ve often wondered if my name alone affected how employers responded to my job applications. Inspired by the seminal 2004 study _"Are Emily and Greg More Employable than Lakisha and Jamal?"_, I wanted to explore how **race- and gender-coded names influence callback rates** using real-world resume data.

The goal is to **detect potential biases in hiring** by analyzing resume content and callback outcomes using machine learning techniques. I trained models **with and without sensitive features** like `firstname` to see how much they influenced prediction and fairness.

### Key Goals:
- Highlight potential discrimination in resume-based screening.
- Explore model fairness with and without proxy features like names.
- Share a personal data science journey rooted in lived experience.
- Empower others to audit algorithms that affect their lives.

---

## Dataset Overview

The dataset is drawn from a real audit study of racial and gender discrimination in hiring. Each row represents a unique resume submitted to a job posting.

**Key Contents:**
- **Resume content**: Skills, work history, honors, gaps, etc.  
- **Applicant features**: First names coded by race/gender  
- **Job post info**: Requirements for education, skills, etc.  
- **Target**: `received_callback` (1 = got a callback; 0 = no callback)

> I removed irrelevant metadata (like job URLs) and cleaned yes/no columns into binary format. I also handled missing values and encoded categorical fields.

**Sensitive proxy variables** like `firstname` were included early in modeling but later removed to evaluate their outsized influence.

---

## Exploratory Data Analysis (EDA)

Before modeling, I performed EDA to detect patterns, correlations, and possible discrimination indicators.

### Key Insights:
- **Gender Bias**: Male-sounding names received more callbacks than female-sounding ones.
- **Racial Bias**: White-sounding names were significantly more likely to receive callbacks than Black-sounding names.
- **Correlated Features**: Degree completion, email presence, and honors slightly correlated with callback likelihood.

---

### Visualizations

#### Callback Rate by Gender
![Callback Rate by Gender](callback%20rate%20gender.png)

**Insight**: Male applicants had a higher callback rate — a clear signal of gender-related bias in hiring responses.

---

#### Callback Rate by Race
![Callback Rate by Race](callback%20rate%20race.png)

**Insight**: Applicants with White-sounding names had a significantly higher callback rate, supporting past research on racial bias.

---

#### Correlation Matrix
![Correlation Matrix](correlation%20matrix.png)

**Insight**:
- Some expected predictors like `college_degree`, `honors`, and `has_email_address` show weak positive correlation.
- Surprisingly, `firstname` correlates more strongly with callback — implying name-based bias.

---

#### Resume Distribution
![Resume Distribution](resume%20distribution.png)

**Insight**: Visualizing the balance of race and gender shows slight imbalance in the dataset. This justifies later use of **SMOTE** for fairer model training.

---

## Column Descriptions

| **Column Name**          | **Description**                                                                |
|--------------------------|--------------------------------------------------------------------------------|
| `received_callback`      | **Target variable**: 1 = received a callback, 0 = no callback                  |
| `firstname`              | Proxy for race and gender (used initially, then removed for fairness testing) |
| `gender`                 | Inferred gender: 1 = Male, 0 = Female                                          |
| `race`                   | Inferred race: 0 = White-sounding, 1 = Black-sounding                          |
| `college_degree`         | Whether the applicant has a college degree                                     |
| `honors`                 | Resume includes honors or awards                                               |
| `worked_during_school`   | Worked while attending school                                                  |
| `volunteer`              | Has volunteer experience                                                       |
| `military`               | Has military experience                                                        |
| `employment_holes`       | Gaps in employment history                                                     |
| `has_email_address`      | Resume includes an email address                                               |
| `job_req_any`            | Job required general experience or "any" skills                                |
| `job_req_communication`  | Job required communication skills                                              |
| `job_req_education`      | Job listed education as a requirement                                          |
| `job_req_computer`       | Computer skills required by the job                                            |
| `job_req_organization`   | Job required organizational skills                                             |
| `job_req_school`         | Mentions of school in job requirements                                         |
| `computer_skills`        | Resume lists computer-related experience                                       |
| `special_skills`         | Resume lists special/technical skills                                          |

---
## Chi-Square Tests for Independence

To statistically assess whether **race** and **gender** significantly affect callback rates, I performed Chi-Square tests for independence.

These tests help answer:  
*Are the differences in callback rates between groups due to chance or actual bias?*


### Gender vs. Callback Rate

```python
from scipy.stats import chi2_contingency

gender_callback_ct = pd.crosstab(df['gender'], df['received_callback'])
chi2_gender, p_gender, dof_gender, _ = chi2_contingency(gender_callback_ct)
print("Chi2 Gender:", chi2_gender)
print("P-value (Gender vs Callback):", p_gender)
```
Result:
If p < 0.05, we reject the null hypothesis and conclude that gender is significantly associated with callback outcomes.

### Race vs. Callback Rate
```python
race_callback_ct = pd.crosstab(df['race'], df['received_callback'])
chi2_race, p_race, dof_race, _ = chi2_contingency(race_callback_ct)
print("Chi2 Race:", chi2_race)
print("P-value (Race vs Callback):", p_race)
```
Result:
If p < 0.05, this means there’s a statistically significant association between race-coded names and callback rates.

### Intrepretation
Result:
If p < 0.05, this means there’s a statistically significant association between race-coded names and callback rates.

**Race and gender significantly impact callback rates,** affirming the presence of bias in hiring decisions.

These statistical findings support the need for **fairness-aware modeling** in the next phase of this project.

---
## Logistic Regression Modeling

To build a baseline model for predicting callback outcomes, I used **Logistic Regression** — a simple, interpretable model ideal for binary classification.

---

### Modeling Pipeline

The pipeline included:

- **One-hot encoding** categorical features
- **Train/test split** (80/20)
- **Class weighting** to handle imbalance
- **Model training and evaluation**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

X = df.drop(columns=['received_callback'])
y = df['received_callback']
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
pipeline.fit(X_train, y_train)
```
## Model Evaluation
```
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print(f"ROC AUC Score: {roc_auc:.4f}")
```
**ROC AUC Score: 0.6226**
This baseline score will serve as a benchmark for fairer models later.

## Visual Results 
![Confusion Matrix – Logistic Regression (Baseline Model)](confusion%20matrix-logical%20regression%20%28baseline%20model%29.png)

![ROC Curve – Logistic Regression](ROC%20curve.png)

- Shows True/False Positives & Negatives
- Area under the curve (AUC) = 0.6226 — moderately better than chance

## Insight 
The logistic regression model leaned heavily on features like `firstname`, a strong proxy for race/gender. This confirmed:

**Sensitive attributes strongly influence predictions.**

That’s why the next step is to test a **fairer model**, removing biased features and rebalancing the data using SMOTE.

---
## Fairness Modeling with SMOTE

After observing bias in our baseline logistic regression, we took a fairness-aware approach:

### Adjustments Made:
- **Removed `firstname`** from the feature set to reduce racial/gender proxy influence.
- **Rebalanced the dataset** using **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance and improve model generalization.

### Visual Results:

![Confusion Matrix – No Firstname](confusion%20matrix%20%28no%20firstname%29.png)

![ROC Curve – No Firstname](random%20forest%20ROC%20curve%20%28no%20firstname%29.png)

### Insights:
- Removing `firstname` caused a **drop in predictive performance** but improved fairness.
- SMOTE helped recover performance by balancing callback vs. non-callback classes.
- The model now **relies more on resume content**, like education, experience, and skills, rather than name proxies.
---
## Random Forest Modeling

To enhance prediction performance and model interpretability, we used a **Random Forest Classifier**.

Random forests are ensemble models that reduce overfitting by combining multiple decision trees, making them ideal for high-dimensional and noisy datasets like resumes.

### Why Random Forest?
- Handles **non-linear relationships** and feature interactions well
- Provides **feature importance scores**
- More robust to noise and outliers than logistic regression

### Performance (With `firstname` Included)

![Confusion Matrix – Random Forest](random%20forest%20confusion%20matrix.png)

![ROC Curve – Random Forest](random%20forest%20ROC%20curve.png)

- **ROC AUC Score**: ~0.70 (better than logistic regression baseline)
- Strong reliance on sensitive features like `firstname`, again confirming bias in real-world hiring

---

### Fair Model: Random Forest (No `firstname` + SMOTE)

We removed `firstname` and applied SMOTE to train a **fairer model**.

![ROC Curve – No Firstname (Random Forest)](random%20forest%20ROC%20curve%20%28no%20firstname%29.png)

- ROC AUC Score dropped slightly, but fairness improved
- Feature importance now highlights relevant resume traits like `college_degree`, `honors`, and `special_skills`

### Insights:
- Random Forest outperformed logistic regression in raw accuracy
- But fairness-aware versions (without `firstname`) perform more ethically
- This reinforces that **biased features drive model performance but at the cost of equity**

We now explore **feature importances** and what drives model decisions.

This stage demonstrates the tradeoff between accuracy and fairness in sensitive decision-making systems like hiring algorithms.

---
## Feature Importance Analysis

After training the Random Forest model (without `firstname`), we examined which features influenced predictions the most.

### Top 10 Important Features

![Top 10 Feature Importances](top%2010%20featured.png)

This visualization highlights the most predictive resume features when bias-prone columns are removed.

### Key Observations:

- `college_degree`, `honors`, and `special_skills` stood out as top indicators of callback likelihood.
- Presence of an `email address` and previous `volunteer` experience were also influential.
- Interestingly, `military` and `worked_during_school` had less importance, showing potential employer bias in evaluating these traits.

### Why This Matters:
Removing biased columns didn't just improve fairness, it revealed what employers *should* value:
- Qualifications
- Experience
- Skills

This step was crucial to shift model decisions toward merit-based evaluation.

---
## Fairness vs. Performance Tradeoff

By removing the `firstname` column and applying SMOTE to balance the dataset, we aimed to reduce bias and improve fairness — but this came with performance tradeoffs.

### ROC Curve Comparison

- **With Firstname (Baseline Model):**  
  ![Baseline ROC Curve](ROC%20curve.png)

- **Without Firstname (Fair Model):**  
  ![ROC Curve Without Firstname](random%20forest%20ROC%20curve%20(no%20firstname).png)

### Confusion Matrix Comparison

- **With Firstname:**  
  ![Confusion Matrix - Baseline](confusion%20matrix-logical%20regression%20(baseline%20model).png)

- **Without Firstname:**  
  ![Confusion Matrix - No Firstname](confusion%20matrix%20(no%20firstname).png)

### Insights:

- Removing `firstname` slightly reduced overall performance (AUC dropped).
- However, **false negatives improved** — more underrepresented candidates were correctly identified.
- This is a common tradeoff in fair ML: **better equity often comes at a cost to raw accuracy.**

### Why This Tradeoff Matters:

While the baseline model seemed "better" on paper, it leaned on sensitive, potentially discriminatory features. The fair model performs more ethically by shifting decision weight toward skills, education, and experience, not names.

Fairness isn't about perfection, it's about improving how we evaluate *people*, not proxies.

---
## Final Thoughts & Reflections

This project started from a place of frustration — sending out resumes, wondering if my name held me back. It turned into something much deeper: a personal data science journey exploring fairness, ethics, and representation in machine learning.

### Key Takeaways

- **Names Matter:** Features like `firstname` can carry deep racial and gender associations, heavily influencing model decisions.
- **Bias is Measurable:** Through statistical tests and model comparisons, we saw how callback disparities exist — even when controlling for education and skills.
- **Fairness Techniques Work:** Removing sensitive proxies and applying SMOTE helped improve model equity, even if performance dropped slightly.
- **Ethics > Accuracy (Sometimes):** A model that prioritizes fairness and minimizes discrimination is more valuable in real-world, human-impacting applications.
  
---
## Issues Encountered

Throughout this project, I faced several challenges that required debugging, reworking code, and learning new tools. These roadblocks helped me grow as a data scientist.

### Data & Cleaning

- **Missing Values:** Many binary columns had inconsistent `yes/no`, `1/0`, or blank values — required manual normalization.
- **Duplicate columns:** Some fields were redundant or proxies for the same attribute.
- **Sensitive Features:** `firstname`, `race`, and `gender` needed careful handling to avoid leakage and unfair conclusions.

### Modeling Issues

- **ROC AUC Score was low (~0.62)** — indicating models struggled to meaningfully distinguish callbacks.
- **Imbalanced Data:** Callback rate was low, causing models to predict the majority class (no callback). This required SMOTE (Synthetic Minority Over-sampling Technique).
- **Bias Dominance:** `firstname` had overwhelming influence on predictions — needed to be dropped to allow fairer signals to emerge.

### Technical Hiccups

- `ModuleNotFoundError: No module named 'imblearn'` when trying to apply SMOTE — fixed with:
  ```python
  !pip install imbalanced-learn
---
### What I Learned

- Practical ML workflows: preprocessing, modeling, tuning, and evaluation
- How to audit models for bias using data science tools
- The balance between raw performance and equitable outcomes
- How to tell a story with data, especially one rooted in lived experience

This project is proof that data science can be more than math. It can reflect the world we live in and help us reshape it.


