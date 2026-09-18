# Family Stress Analysis System

## Project Overview

The **Family Stress Analysis System** is a web-based data mining application developed to analyze factors associated with family stress and predict a user's stress category.

The project applies both **descriptive data mining** and **predictive data mining** techniques. A Django-based web interface allows users to enter personal, family, employment, and lifestyle information and receive a predicted stress category.

## Objectives

* Analyze important factors associated with family stress.
* Discover relationships and patterns in the dataset.
* Identify frequent combinations of factors associated with high stress.
* Build a machine learning model to predict stress categories.
* Provide a simple web-based interface for stress prediction.

## Dataset

The original dataset contains **20,100 records** and **17 attributes**.

After data preprocessing and cleaning, **18,419 records** were used for analysis.

The target variable is:

* **Low**
* **Medium**
* **High**

The `Stress_level` attribute was removed before model training to avoid target leakage.

## Data Mining Methods

### Descriptive Mining

The project uses:

* K-Means Clustering
* Apriori Association Rule Mining
* Statistical association analysis

Important findings include:

* Work-family balance was an important numerical factor associated with stress.
* Marital status showed the strongest association among the categorical variables examined.
* Association rules identified combinations involving low work-family balance, family time, income, and number of children that were associated with high stress.

### Predictive Mining

Three classification approaches were evaluated:

* Dummy Classifier
* Logistic Regression
* Random Forest Classifier

The final Random Forest model was trained using an **80/20 stratified train-test split**.

## Model Performance

The final Random Forest model achieved:

| Metric                       | Result |
| ---------------------------- | -----: |
| Accuracy                     | 84.85% |
| Weighted Precision           | 84.63% |
| Weighted Recall              | 84.85% |
| Weighted F1-Score            | 84.37% |
| 5-Fold Cross-Validation Mean | 84.67% |

The model predicts three stress categories:

**High, Low, and Medium**

## Technologies Used

* Python
* Django
* Pandas
* NumPy
* Scikit-learn
* Joblib
* HTML
* CSS
* JavaScript
* Git
* GitHub
* Git LFS

## Project Structure

```text
family-stress-analysis/
│
├── family_stress_system/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── models/
│   ├── model_features.pkl
│   └── random_forest_model.pkl
│
├── stress/
│   ├── migrations/
│   ├── templates/
│   │   └── stress/
│   │       ├── home.html
│   │       ├── dashboard.html
│   │       └── prediction.html
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
│
├── manage.py
├── test_model.py
├── .gitignore
└── .gitattributes
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YoonNandarMoe17/family-stress-analysis.git
```

Then enter the project folder:

```bash
cd family-stress-analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

## Running the System

Run Django migrations:

```bash
python manage.py migrate
```

Start the development server:

```bash
python manage.py runserver
```

Open the following address in a web browser:

```text
http://127.0.0.1:8000/
```

## Using the System

1. Open the home page.
2. Navigate to the stress prediction page.
3. Enter the required personal, family, employment, and lifestyle information.
4. Submit the form.
5. The system processes the input using the trained Random Forest model.
6. The predicted stress category is displayed.

## Machine Learning Model

The trained Random Forest model is stored using **Git LFS** because the model file is larger than GitHub's normal web-upload limit.

The model file is:

```text
models/random_forest_model.pkl
```

The feature information used by the model is stored in:

```text
models/model_features.pkl
```

## Project Repository

**GitHub Repository:**

https://github.com/YoonNandarMoe17/family-stress-analysis

## Project Information

**Project:** Family Stress Analysis Project
**Course:** IS-212 Data and Knowledge Mining
**University:** University of Computer Studies, Yangon
**Application Type:** Django Web Application
**Machine Learning Model:** Random Forest Classifier
