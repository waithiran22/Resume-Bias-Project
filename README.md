# Resume Bias Project
## About This Project

This project is deeply personal. As a Black woman named **Waithira**, I often wondered if my name alone influenced how employers responded to my job applications. Inspired by studies like the 2004 "Are Emily and Greg More Employable than Lakisha and Jamal?" paper, I wanted to explore how race- and gender-coded names affect callback rates using real resume data.

The goal is to **detect potential biases in hiring** by analyzing resume features and callback outcomes using machine learning. I trained models with and without sensitive features (like `firstname`) to see how much they influence predictions.

**Key goals of this project:**
- Highlight potential discrimination in hiring processes.
- Explore fairness in model performance with and without proxy features.
- Share a personal data science journey rooted in real-world pain and curiosity.
- Empower others to audit algorithms and data that affect their lives.

## Dataset Overview

The dataset used in this project comes from a real-world audit study of racial and gender discrimination in hiring. Each row represents a submitted resume and includes:

- **Resume content** (skills, education, work history)
- **Applicant features** (race-coded and gender-coded via names)
- **Job posting details**
- **Whether the resume received a callback (target variable)**

To preserve integrity, I removed unnecessary columns like job links and IDs, and cleaned binary columns (`yes/no`) into numerical format. Sensitive fields like `firstname`, which can proxy for race and gender, were analyzed but later excluded to measure bias impact.

**Target variable**: `received_callback` (1 if the applicant got a callback, 0 otherwise)

This dataset allowed me to:
- Explore how features correlate with callback rates
- Measure racial/gender bias with statistical tests
- Build predictive models and observe fairness

## Exploratory Data Analysis (EDA)

Before modeling, I visualized the data to understand patterns and biases.

### Key Insights:
- **Gender Bias**: Callback rate was higher for male-sounding names.
- **Racial Bias**: Applicants with White-sounding names received significantly more callbacks than Black-sounding names.
- **Feature Correlations**: Education, email presence, and honors had some correlation with callbacks.

### Visuals Included:
-  `callback_by_gender.png`
-  `callback_by_race.png`
-  `correlation_matrix.png`
-  
![Callback Rate by Gender](visualizations/callback_by_gender.png)
![Callback Rate by Race](visualizations/callback_by_race.png)
![Correlation Matrix](visualizations/correlation_matrix.png)

These visuals were critical in validating the need for fairness-aware modeling.

## Column Descriptions

| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `received_callback`     | Target variable - 1 if applicant received a callback, 0 otherwise           |
| `firstname`             | First name of applicant (proxy for race/gender, used for bias analysis)    |
| `gender`                | Gender coded as binary (1 = Male, 0 = Female)                              |
| `race`                  | Race coded from name (0 = White-sounding, 1 = Black-sounding)               |
| `college_degree`        | Whether applicant holds a college degree (1 = Yes, 0 = No)                  |
| `honors`                | Whether honors are listed on resume                                         |
| `worked_during_school`  | Experience working during education                                         |
| `volunteer`             | Volunteer experience indicated                                              |
| `military`              | Military experience                                                         |
| `employment_holes`      | Gaps in employment history                                                  |
| `has_email_address`     | Presence of email address on resume                                         |
| `job_req_any`           | Job required any skills                                                     |
| `job_req_communication` | Communication skills required                                               |
| `job_req_education`     | Education listed as a requirement                                           |
| `job_req_computer`      | Computer skills required                                                    |
| `job_req_organization`  | Organizational skills required                                              |
| `job_req_school`        | Schooling mentioned in job requirements                                     |
| `computer_skills`       | Resume lists computer-related skills                                        |
| `special_skills`        | Resume lists special/technical skills                                       |

