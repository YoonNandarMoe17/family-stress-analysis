from django.shortcuts import render
import joblib
import os
import pandas as pd


# ==============================
# Load Trained Random Forest Model
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

features_path = os.path.join(
    BASE_DIR,
    "models",
    "model_features.pkl"
)

rf_model = joblib.load(model_path)
model_features = joblib.load(features_path)


# ==============================
# Home Page
# ==============================

def home(request):

    return render(
        request,
        "stress/home.html"
    )


# ==============================
# Prediction Page
# ==============================

def prediction(request):

    prediction_result = None

    if request.method == "POST":

        # ==============================
        # Get User Input
        # ==============================

        data = {

            "Age": float(
                request.POST.get("age")
            ),

            "Gender": request.POST.get(
                "gender"
            ).lower(),

            "Education_level": request.POST.get(
                "education_level"
            ).lower(),

            "Employment_status": request.POST.get(
                "employment_status"
            ).lower(),

            "Family_history_mental_illness":
                request.POST.get(
                    "family_history_mental_illness"
                ).lower(),

            "Annual_personal_income": float(
                request.POST.get(
                    "annual_personal_income"
                )
            ),

            "Annual_family_income": float(
                request.POST.get(
                    "annual_family_income"
                )
            ),

            "Relationship_with_family":
                request.POST.get(
                    "relationship_with_family"
                ).lower(),

            "Marital_status": request.POST.get(
                "marital_status"
            ).lower(),

            "Number_of_children": float(
                request.POST.get(
                    "number_of_children"
                )
            ),

            "Number_of_family_members": float(
                request.POST.get(
                    "number_of_family_members"
                )
            ),

            "Job_type": request.POST.get(
                "job_type"
            ).lower(),

            "Employment_contract_type":
                request.POST.get(
                    "employment_contract_type"
                ).lower(),

            "Daily_time_dedicated_to_family":
                float(
                    request.POST.get(
                        "daily_time_dedicated_to_family"
                    )
                ),

            "Work_family_balance":
                float(
                    request.POST.get(
                        "work_family_balance"
                    )
                ),
        }


        # ==============================
        # Convert Input to DataFrame
        # ==============================

        input_df = pd.DataFrame([data])


        print("\n========================================")
        print("         ORIGINAL USER INPUT")
        print("========================================")
        print(input_df)


        # ==============================
        # One-Hot Encoding
        # ==============================

        input_df = pd.get_dummies(
            input_df,
            drop_first=False
        )


        print("\n========================================")
        print("       AFTER ONE-HOT ENCODING")
        print("========================================")
        print(input_df)


        # ==============================
        # Match Model Features
        # ==============================

        input_df = input_df.reindex(
            columns=model_features,
            fill_value=0
        )


        print("\n========================================")
        print("       MODEL INPUT AFTER REINDEX")
        print("========================================")
        print(input_df)


        # ==============================
        # Model Information
        # ==============================

        print("\n========================================")
        print("          MODEL INFORMATION")
        print("========================================")

        print(
            "Number of model features:",
            len(model_features)
        )

        print(
            "Input shape:",
            input_df.shape
        )

        print(
            "Model expected features:",
            rf_model.n_features_in_
        )


        # ==============================
        # Non-Zero Features
        # ==============================

        print("\n========================================")
        print("          NON-ZERO FEATURES")
        print("========================================")

        non_zero_features = input_df.loc[
            :,
            (input_df != 0).any(axis=0)
        ]

        print(non_zero_features)


        # ==============================
        # Prediction
        # ==============================

        prediction_result = rf_model.predict(
            input_df
        )[0]


        # ==============================
        # Probabilities
        # ==============================

        probabilities = rf_model.predict_proba(
            input_df
        )[0]


        print("\n========================================")
        print("             FINAL RESULT")
        print("========================================")

        print(
            "Prediction:",
            prediction_result
        )

        print(
            "Probabilities:",
            probabilities
        )

        print(
            "Classes:",
            rf_model.classes_
        )

        print("========================================\n")


    return render(
        request,
        "stress/prediction.html",
        {
            "prediction": prediction_result
        }
    )


# ==============================
# Dashboard Page
# ==============================

def dashboard(request):

    return render(
        request,
        "stress/dashboard.html"
    )