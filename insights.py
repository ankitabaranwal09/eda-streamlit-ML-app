def generate_insights(df):
    insights = []

    # Missing values
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            insights.append(f"{col} has {val} missing values")

    # High correlation
    corr = df.select_dtypes(include='number').corr()
    print(corr)
    for col in corr.columns:
        for idx in corr.index:
            if col != idx and abs(corr.loc[col, idx]) > 0.7:
                insights.append(f"{col} and {idx} are highly correlated")

    # Outliers (simple rule)
    for col in df.select_dtypes(include='number').columns:
        if df[col].std() > df[col].mean():
            insights.append(f"{col} may contain outliers")

    # ✅ ADD THIS
    if not insights:
        insights.append("Dataset looks clean. No major issues detected.")

    return insights



