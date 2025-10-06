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

## Interactive Tableau Dashboard   
You can download and open the interactive Tableau workbook here:  

[Bias in Resume Callbacks (Tableau Workbook)](Bias%20in%20resume%20callbacks.twbx)

The dashboard includes:  
- **Bias in Callbacks** → Callback rates by race and gender.  
- **Model Performance** → ROC curves + Confusion Matrices with toggle between Logistic Regression and Random Forest.  
- **What Employers Value** → Feature importance rankings once biased features are removed.  
- **Resume Correlations** → Heatmap of education, honors, gaps, and skills.  

![Dashboard Snapshot](Dashboard%20Image.png)

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

Visual examples:  
![Callback Rate by Race](callback%20rate%20race.png)
  
![Callback Rate by Gender](callback%20rate%20gender.png)
  
![Confusion Matrix – Logistic Regression](confusion%20matrix-logical%20regression%20(baseline%20model).png)

---
## Statistical Testing (Chi-Square)

To check whether differences in callback rates were random or evidence of bias, I ran Chi-Square independence tests:

-Gender vs Callback → significant (p < 0.05)
-Race vs Callback → significant (p < 0.05)

Result: Callbacks are significantly associated with both race and gender, meaning disparities are not due to chance but evidence of real bias

---
## Baseline Model: Logistic Regression
I built a Logistic Regression model to predict callbacks.

 -ROC AUC:0.62 (moderately better than chance)
*-Issue: The model leaned heavily on firstname (a strong proxy for race/gender).*

-Insight: Even algorithms replicate bias — the model “learned” that names matter more than skills or education.

---
### Fair Model with SMOTE (No firstname)  
- Dropped firstname.  
- Balanced dataset with SMOTE.  
- Improved fairness, reduced reliance on demographic proxies.  

### Random Forest  
- AUC: ~0.70 with firstname, dropped when removed.  
- Feature importances highlighted **college_degree**, **honors**, **special_skills**.  

ROC Example:  
![](random%20forest%20ROC%20curve.png)  

Confusion Matrix Example:  
![](confusion%20matrix%20(no%20firstname).png)  

---

## 🔑 Key Takeaways  
- **Names Matter** → `firstname` strongly drives predictions, confirming bias.  
- **Bias is Measurable** → Statistical tests and ML confirm disparities.  
- **Fairness Techniques Work** → Removing proxy features + SMOTE improves equity.  
- **Ethics > Accuracy (Sometimes)** → A fairer model with slightly lower accuracy is better for real-world use.  

---

## Reflections  
This project started as frustration, sending resumes and wondering if my name held me back — but became a journey into **fairness, ethics, and representation** in data science.  

I learned how to:  
- Audit models for bias.  
- Balance **performance vs. fairness** tradeoffs.  
- Tell a story with data rooted in lived experience.  


### Data & Cleaning

- **Missing Values:** Many binary columns had inconsistent `yes/no`, `1/0`, or blank values — required manual normalization.
- **Duplicate columns:** Some fields were redundant or proxies for the same attribute.
- **Sensitive Features:** `firstname`, `race`, and `gender` needed careful handling to avoid leakage and unfair conclusions.

### Modeling Issues

- **ROC AUC Score was low (~0.62)** — indicating models struggled to meaningfully distinguish callbacks.
- **Imbalanced Data:** Callback rate was low, causing models to predict the majority class (no callback). This required SMOTE (Synthetic Minority Over-sampling Technique).
- **Bias Dominance:** `firstname` had o
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
verwhelming influence on predictions — needed to be dropped to allow fairer signals to emerge.

This project is proof that data science can be more than math. It can reflect the world we live in and help us reshape it.


