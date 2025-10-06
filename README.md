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
  
**Sensitive proxy variables** like `firstname` were included early in modeling but later removed to evaluate their outsized influence.

---

## Exploratory Data Analysis (EDA)

Before modeling, I performed EDA to detect patterns, correlations, and possible discrimination indicators.

---
![Callback Rate by Race](https://github.com/waithiran22/Resume-Bias-Project/blob/main/Callback%20Rate%20by%20Race.png?raw=true)
  
![Callback Rate by Gender](https://github.com/waithiran22/Resume-Bias-Project/blob/main/Callback%20Rate%20by%20Gender.png?raw=true)
  
![Correlation Heatmap](https://github.com/waithiran22/Resume-Bias-Project/blob/main/Correlation%20Heatmap.png?raw=true)

---

## Statistical Testing (Chi-Square)

To check whether differences in callback rates were random or evidence of bias, I ran Chi-Square independence tests:

-Gender vs Callback → significant (p < 0.05)

-Race vs Callback → significant (p < 0.05)

*-Result: Callbacks are significantly associated with both race and gender, meaning disparities are not due to chance but evidence of real bias*

---

## Baseline Model: Logistic Regression
-ROC AUC:0.62 (moderately better than chance)

*-Issue: The model leaned heavily on firstname (a strong proxy for race/gender).*

![ROC Curve (Model Performance)](https://github.com/waithiran22/Resume-Bias-Project/blob/main/ROC%20Curve%20(Model%20Performance).png?raw=true)

![Confusion Matrix](https://github.com/waithiran22/Resume-Bias-Project/blob/main/Confusion%20Matrix.png?raw=true)

---

### Fairness-Aware Modeling 
- Dropped firstname.  
- Balanced dataset with SMOTE.  
- Improved fairness, reduced reliance on demographic proxies.  

### Random Forest  
- AUC: ~0.70 with firstname, dropped when removed.  
- Feature importances highlighted **college_degree**, **honors**, **special_skills**.  

![Feature Correlation Heatmap](https://github.com/waithiran22/Resume-Bias-Project/blob/main/Feature%20Correlation%20Heatmap.png?raw=true)
Insight: Removing biased features revealed what employers should value: education, skill sand experience

---

## Key Takeaways  
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

---
### Data & Cleaning

- Missing Values: Many binary columns had inconsistent formats (yes/no, 1/0, or blanks). These were normalized into consistent binary values.

- Duplicate Columns: Some features were redundant or served as proxies for the same attribute, so I removed them to avoid double-counting.

- Sensitive Features: Variables like firstname, race, and gender needed careful handling. They were included initially for bias detection but later excluded to avoid unfair leakage into model predictions.

### Modeling Challenges

-Low ROC AUC (~0.62): Early models struggled to meaningfully distinguish callbacks from non-callbacks.

-Imbalanced Data: Callback rates were very low, so the models tended to predict the majority class (no callback). To address this, I used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

-Bias Dominance: The firstname feature had overwhelming influence on predictions, confirming that name-based bias was being replicated by the models. This feature was dropped in fairness-aware versions.

---

### Technical Hiccups


- `ModuleNotFoundError: No module named 'imblearn'` when trying to apply SMOTE — fixed with:
  ```python
  !pip install imbalanced-learn
  
---

### What I Learned

- Practical ML workflows: preprocessing, modeling, tuning, and evaluation
- How to audit models for bias using data science tools
- The balance between raw performance and equitable outcomes
