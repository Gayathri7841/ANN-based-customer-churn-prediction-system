import tensorflow as tf
import pickle
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model("model.h5")


# ============================================================
# 2. LOAD SAVED PREPROCESSORS
# ============================================================
    
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# 3. STREAMLIT TITLE
# ============================================================

st.title("Customer Churn Prediction")

st.write(
    "Enter customer details below to predict whether the customer "
    "is likely to churn."
)


# ============================================================
# 4. USER INPUTS
# ============================================================

geography = st.selectbox(
    "Geography",
    one_hot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=40
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# ============================================================
# 5. PREDICT BUTTON
# ============================================================

if st.button("Predict Churn"):

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }


    # --------------------------------------------------------
    # Convert dictionary to DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])


    # ========================================================
    # 6. ENCODE GENDER
    # ========================================================

    input_df["Gender"] = label_encoder_gender.transform(
        input_df["Gender"]
    )


    # ========================================================
    # 7. ONE-HOT ENCODE GEOGRAPHY
    # ========================================================

    geo_encoded = one_hot_encoder_geo.transform(
        input_df[["Geography"]]
    )


    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geo.get_feature_names_out(
            ["Geography"]
        ),
        index=input_df.index
    )


    # --------------------------------------------------------
    # Remove original Geography column
    # --------------------------------------------------------

    input_df = input_df.drop(
        "Geography",
        axis=1
    )


    # --------------------------------------------------------
    # Add encoded Geography columns
    # --------------------------------------------------------

    input_df = pd.concat(
        [
            input_df,
            geo_encoded_df
        ],
        axis=1
    )


    # ========================================================
    # 8. ARRANGE FEATURES IN TRAINING ORDER
    # ========================================================

    input_df = input_df[
        scaler.feature_names_in_
    ]


    # ========================================================
    # 9. SCALE INPUT
    # ========================================================

    input_scaled = scaler.transform(
        input_df
    )


    # ========================================================
    # 10. MAKE PREDICTION
    # ========================================================

    prediction_probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # Convert probability to percentage

    probability = prediction_probability * 100


    # ========================================================
    # 11. DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")


    st.metric(
        "Churn Probability",
        f"{probability:.2f}%"
    )


    # ========================================================
    # 12. CHURN / NO CHURN
    # ========================================================

    if prediction_probability >= 0.5:

        st.error(
            "⚠️ Customer is likely to churn"
        )

        st.write(
            "The model predicts that this customer may leave the bank."
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )

        st.write(
            "The model predicts that this customer is likely to remain with the bank."
        )