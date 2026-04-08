import pandas as pd
import numpy as np


def load_data(file):
    import pandas as pd

    file.seek(0)
    content = file.read()

    from io import StringIO
    data = StringIO(content.decode('utf-8', errors='ignore'))

    return pd.read_csv(data)    
            
def get_columns(df):
    numeric = df.select_dtypes(include = 'number').columns.tolist()
    categorical = df.select_dtypes(include = 'object').columns.tolist()
    return numeric, categorical

def handle_missing_values(df):
    df = df.copy()
    report = []

    for col in list(df.columns):
        null_percent = df[col].isnull().mean() * 100

        # DROP if > 75% missing (for ALL columns)
        if null_percent > 75:
            df.drop(columns=[col], inplace=True)
            report.append(f"{col}: Dropped ({round(null_percent,2)}% missing)")
            continue

        # NUMERIC
        if df[col].dtype in ['int64', 'float64']:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                report.append(f"{col}: Filled with median ({median_val})")

        # CATEGORICAL
        else:
            if df[col].isnull().sum() > 0:
                mode_series = df[col].mode()
                if not mode_series.empty:
                    mode_val = mode_series[0]
                    df[col].fillna(mode_val, inplace = True)
                    report.append(f"{col}: filled with mode ({mode_val})")

    return df, report

# Duplicate Rows 
def remove_duplicate_rows(df):
    report = []
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    
    if before != after:
        report.append(f"Removed {before - after} duplicate rows")
    else:
        report.append("No duplicate rows found")
    return df, report
    
# Duplicate columns
def remove_duplicate_columns(df):
    report = []
    before = df.shape[1]
    df = df.loc[:, ~df.T.duplicated()]
    after = df.shape[1]
    if before != after:
        report.append(f"Removed {before - after} duplicate columns")
    else:
        report.append("No duplicate columns found")
    return df, report


def handle_outliers(df, target_col=None, method="iqr", action="cap"):
    df = df.copy()
    report = []

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        # 🔥 SKIP TARGET COLUMN
        if col == target_col:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if outliers > 0:
            if action == "remove":
                before = df.shape[0]
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                after = df.shape[0]
                report.append(f"{col}: Removed {before - after} rows")

            elif action == "cap":
                df[col] = df[col].clip(lower, upper)
                report.append(f"{col}: Capped {outliers} outliers")

        else:
            report.append(f"{col}: No outliers found")

    return df, report



def clean_data(df, target_col=None, outlier_action="None"):
    report = []

    # Missing values
    df, rep1 = handle_missing_values(df)
    report.extend(rep1)

    # Duplicate rows
    df, rep2 = remove_duplicate_rows(df)
    report.extend(rep2)

    # Duplicate columns
    df, rep3 = remove_duplicate_columns(df)
    report.extend(rep3)

    # 🔥 Outlier handling (SAFE)
    if outlier_action != "None":
        df, rep4 = handle_outliers(
            df,
            target_col=target_col,   # ✅ PASS TARGET HERE
            action=outlier_action.lower()
        )
        report.extend(rep4)
    else:
        report.append("Outlier handling skipped")

    return df, report


def fix_dataframe_types(df):
    import pandas as pd
    import numpy as np

    df = df.copy()

    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64")
            continue

        try:
            converted = pd.to_numeric(df[col])
            if converted.notna().sum() > 0.8 * len(df):
                df[col] = converted
                continue
        except:
            pass

        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > 0.8 * len(df):
                df[col] = converted
                continue
        except:
            pass

        df[col] = df[col].astype(str)

    return df