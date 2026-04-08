import pandas as pd

def make_df_safe(df):
    import numpy as np
    
    safe_df = df.copy()
    safe_df = safe_df.replace([np.inf, -np.inf], np.nan)
    return safe_df

def preprocess_data(X, y, model_choice):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # Handle categorical features
    X = pd.get_dummies(X)

    # Encode target if classification
    le = None
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
    return X, y, le

def basic_info(df):
    return df.shape, df.columns.tolist(), df.dtypes

def summary_stats(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    # ✅ Step 1: Convert possible numeric strings to numbers
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # ✅ Step 2: Detect numeric columns (robust way)
    numeric_df = df.select_dtypes(include=[np.number])

    # ✅ Step 3: Handle edge case
    if numeric_df.shape[1] == 0:
        return pd.DataFrame({"Message": ["⚠️ No numeric columns found"]})

    # ✅ Step 4: Return clean summary
    summary = numeric_df.describe().T

    return summary

def missing_values(df):
    return df.isnull().sum().sort_values(ascending = False)

def correlation_matrix(df):
    return df.select_dtypes(include = 'number').corr()

# ------------ Model Function ---------------
def get_model(model_choice, task, params=None):

    params = params or {}
    if model_choice == "Random Forest":
        if task == "Classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)

    elif model_choice == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)

    elif model_choice == "XGBoost":
        if task == "Classification":
            from xgboost import XGBClassifier
            return XGBClassifier(
                eval_metric='logloss',
                **params
            )
        else:
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
    elif model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            **params
        )