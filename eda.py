import pandas as pd

def make_df_safe(df):
    import numpy as np
    
    safe_df = df.copy()
    safe_df = safe_df.replace([np.inf, -np.inf], np.nan)
    return safe_df

def preprocess_data(X, y, model_choice, task):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # ---------------- X Processing ----------------
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # ---------------- y Processing ----------------
    le = None
    y = pd.Series(y)

    # Remove bad values
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.fillna(0)

    if task == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(y)   # 🔥 ALWAYS encode

        y = pd.Series(y).astype(int)

        if pd.Series(y).nunique() <= 1:
            raise ValueError("Target has only one class")

        # 🔴 CRITICAL CHECK
        if pd.Series(y).nunique() <= 1:
            raise ValueError("Target has only one class")

    else:
        # Regression
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(0)

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
                n_estimators = 100,
                max_depth = 5,
                learning_rate = 0.1,
                eval_metric='logloss',
                tree_method = "hist",
                enable_categorical = False,
                n_jobs = -1,
                use_label_encoder = False,
                **params
            )
        else:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators = 100,
                max_depth = 5,
                learning_rate = 0.1,
                **params
            )
    elif model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            **params
        )