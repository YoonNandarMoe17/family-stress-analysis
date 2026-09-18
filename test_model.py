import joblib
import pandas as pd

# ==============================
# 1. LOAD MODEL
# ==============================

model = joblib.load(
    "models/random_forest_model.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)
print("Number of features:", model.n_features_in_)
print("Number of classes:", len(model.classes_))
print("Classes:", model.classes_)


# ==============================
# 2. FUNCTION TO TEST ONE PERSON
# ==============================

def test_person(name, data):

    df = pd.DataFrame([data])

    # Convert categorical values to lowercase
    categorical_columns = [
        "Gender",
        "Education_level",
        "Employment_status",
        "Family_history_mental_illness",
        "Relationship_with_family",
        "Marital_status",
        "Job_type",
        "Employment_contract_type"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype(str).str.lower()

    # One-hot encode
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns
    )

    # Make exactly the same 45 features as training
    df_encoded = df_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(df_encoded)[0]
    probabilities = model.predict_proba(df_encoded)[0]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Prediction:", prediction)

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"{class_name}: {probability * 100:.2f}%"
        )


# ==============================
# 3. TEST PERSON 1
# ==============================

test_person(
    "Person 1 - Lower Stress Situation",
    {
        "Age": 25,
        "Annual_personal_income": 6000000,
        "Annual_family_income": 10000000,
        "Number_of_children": 0,
        "Number_of_family_members": 3,
        "Daily_time_dedicated_to_family": 5,
        "Work_family_balance": 5,

        "Gender": "female",
        "Education_level": "bachelor's",
        "Employment_status": "employed",
        "Family_history_mental_illness": "no",
        "Relationship_with_family": "excellent",
        "Marital_status": "single",
        "Job_type": "employee",
        "Employment_contract_type": "permanent"
    }
)


# ==============================
# 4. TEST PERSON 2
# ==============================

test_person(
    "Person 2 - Higher Stress Situation",
    {
        "Age": 45,
        "Annual_personal_income": 2000000,
        "Annual_family_income": 3000000,
        "Number_of_children": 4,
        "Number_of_family_members": 6,
        "Daily_time_dedicated_to_family": 1,
        "Work_family_balance": 1,

        "Gender": "male",
        "Education_level": "high school",
        "Employment_status": "unemployed",
        "Family_history_mental_illness": "yes",
        "Relationship_with_family": "bad",
        "Marital_status": "married",
        "Job_type": "worker",
        "Employment_contract_type": "temporary"
    }
)


# ==============================
# 5. TEST PERSON 3
# ==============================

test_person(
    "Person 3 - Moderate Situation",
    {
        "Age": 35,
        "Annual_personal_income": 4000000,
        "Annual_family_income": 6000000,
        "Number_of_children": 2,
        "Number_of_family_members": 4,
        "Daily_time_dedicated_to_family": 3,
        "Work_family_balance": 3,

        "Gender": "female",
        "Education_level": "master's",
        "Employment_status": "employed",
        "Family_history_mental_illness": "no",
        "Relationship_with_family": "good",
        "Marital_status": "married",
        "Job_type": "employee",
        "Employment_contract_type": "permanent"
    }
)