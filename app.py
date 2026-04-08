import streamlit as st
from eda import basic_info, summary_stats, missing_values, make_df_safe, preprocess_data, get_model
from insights import generate_insights
from plots import plot_graph, correlation_heatmap
from utils import load_data, get_columns, clean_data, fix_dataframe_types
from langchain_groq import ChatGroq
import pandas as pd
import numpy as np
import os

groq_key = st.secrets["GROQ_API_KEY"]

if not groq_key:
    st.error("❌ API key not found")
    st.stop()

if "scaler" not in st.session_state:
    st.session_state.scaler = None
st.set_page_config(page_title="Smart EDA App", layout="wide")
st.title("📊 AI-Powered Smart EDA & Predictive Analytics Platform")

# Sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_cleaned = st.sidebar.toggle("Use Cleaned Data")

# ---------------- MAIN ----------------
if uploaded_file:

    try:
        import numpy as np
        df = load_data(uploaded_file)
        

    except Exception as e:
        st.error(str(e))
        st.stop()

    # Initialize session state
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.base_df = df.copy()
        st.session_state.original_df = df.copy()
        st.session_state.cleaned_df = None
        st.session_state.cleaning_report = None
        st.session_state.last_file = uploaded_file.name

    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    # -------------------------------
    # 🗑️ STEP 1: DROP COLUMNS
    # -------------------------------
    st.sidebar.subheader("🗑️ Drop Columns")

    drop_cols = st.sidebar.multiselect(
        "Select Columns", st.session_state.original_df.columns
    )

    if st.sidebar.button("Apply Column Drop"):
        if drop_cols:
            st.session_state.original_df = st.session_state.original_df.drop(
                columns=drop_cols, errors='ignore'
            )
            st.sidebar.success(f"✅ Dropped {len(drop_cols)} columns")
        else:
            st.sidebar.warning("⚠️ No columns selected")

    # -------------------------------
    # 🧹 STEP 2: OUTLIER HANDLING
    # -------------------------------
    st.sidebar.subheader("🧹 Outlier Handling")

    outlier_action = st.sidebar.selectbox(
        "Choose Method", ["None", "Remove", "Cap"]
    )

    if outlier_action == "Cap":
        st.sidebar.info("📌 Capping replaces extreme values using Q1 and Q3.")
    elif outlier_action == "Remove":
        st.sidebar.warning("⚠️ Outliers will be removed.")
    else:
        st.sidebar.success("✅ No outlier handling applied.")

    # -------------------------------
    # 🧼 STEP 3: CLEAN DATA
    # -------------------------------
    
    # Step 1: Select target column before Cleanning
    st.sidebar.subheader("🎯 Target Selection")
    target_for_cleaning = st.sidebar.selectbox(
        "🎯 Select Target Column (for safe cleaning)",
        st.session_state.original_df.columns,
        key="clean_target"
    )
    # Step 2: clean Data Button
    if st.sidebar.button("🧹 Clean Data"):
        cleaned_df, report = clean_data(
            st.session_state.original_df,
            outlier_action=outlier_action
        )
        
        # Step 3: Safety Check 
        if cleaned_df[target_for_cleaning].nunique() <= 1:
            st.error("❌ Target column collapsed! Try using 'Cap' instead of 'Remove'.")
        st.session_state.cleaned_df = cleaned_df
        st.session_state.cleaning_report = report

    # -------------------------------
    # 📥 DOWNLOAD
    # -------------------------------
    if st.session_state.cleaned_df is not None:
        st.sidebar.download_button(
            label="📥 Download Cleaned Data",
            data=st.session_state.cleaned_df.astype(str).to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # -------------------------------
    # 🔁 DATA SELECTION
    # -------------------------------
    if use_cleaned and st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        st.sidebar.success("Using Cleaned Data")
    else:
        df = st.session_state.original_df
        st.sidebar.info("Using Original Data")

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["Preview", "EDA", "Visuals", "Insights", "Summary", "ML Model", "AI Assistant","Model Comparison", "Prediction"]
    )

    # -------- TAB 1 --------
    with tab1:
        st.subheader("📄 Dataset Preview")
        st.dataframe(make_df_safe(df), width="stretch")

    # -------- TAB 2 --------
    with tab2:
        st.subheader("📌 Basic Info")

        shape, cols, dtypes = basic_info(df)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Shape")
            st.write(shape)


        with col2:
            st.subheader("📂 Columns")
            with st.expander("View Columns"):
                st.write(cols)

        st.subheader("🧾 Data Types")
        with st.expander("View Data Types"):
            st.dataframe(dtypes.astype(str), width="stretch")
            
        
        st.subheader("📊 Data Quality Score")

        # ------------------- Metrics -------------------
        missing_ratio = df.isnull().sum().sum() / df.size
        duplicate_ratio = df.duplicated().sum() / len(df)
        # Constant columns (no variance)
        constant_cols = df.nunique() == 1
        constant_ratio = constant_cols.sum() / len(df.columns)
        # Infinite values
        import numpy as np
        inf_ratio = np.isinf(df.select_dtypes(include=[np.number])).sum().sum() / df.size
        # ------------------- Scoring -------------------
        score = 100
        score -= missing_ratio * 40 * 100
        score -= duplicate_ratio * 30 * 100
        score -= constant_ratio * 20 * 100
        score -= inf_ratio * 10 * 100
        score = max(score, 0)
        # ------------------- Display -------------------
        st.metric("Score", f"{score:.1f} / 100")

        # ------------------- Feedback -------------------
        if score > 85:
            st.success("✅ Excellent Data Quality")
        elif score > 70:
            st.info("👍 Good Data, minor cleaning needed")
        elif score > 50:
            st.warning("⚠️ Moderate issues, needs cleaning")
        else:
            st.error("❌ Poor Data Quality, major fixes required")

        # Unique values
        st.subheader("🔍 Unique Values per Column")
        unique_df = df.nunique().reset_index()
        unique_df.columns = ["Column", "Unique Values"]
        
        with st.expander("View Unique Values"):
            st.dataframe(unique_df.astype(str), width="stretch")

        # Duplicates
        st.subheader("🔁 Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.metric("Total Duplicate Rows", duplicate_count)
        
        if duplicate_count > 0:
            with st.expander("View Duplicate Rows"):
                st.dataframe(df[df.duplicated()].astype(str), width="stretch")
            st.warning("⚠️ Dataset contains duplicate rows")
        else:
            st.success("✅ No duplicate rows found")
            
        # Summary
        st.subheader("📊 Summary Statistics")
        import pandas as pd
        # ✅ Fix numeric types BEFORE summary
        df_fixed = df.copy()
        for col in df_fixed.columns:
            try:
                df_fixed[col] = pd.to_numeric(df_fixed[col])
            except:
                pass
        safe_summary = summary_stats(df_fixed)

        # ✅ Handle case when no numeric columns
        if isinstance(safe_summary, pd.DataFrame) and safe_summary.shape[1] > 0:
            st.dataframe(safe_summary.astype(str), width="stretch")
        else:
            st.warning("⚠️ No numeric columns found")
            
        # Target Distributuion
        st.subheader("🎯 Target Distribution")
        target_col = st.selectbox("Select Target Column", df.columns)
        chart_data = df[target_col].value_counts()
        st.bar_chart(chart_data)
        
        st.divider()

        # Missing Values
        st.subheader("❗ Missing Values")
        mv = missing_values(df)
        if hasattr(mv, "to_frame"):
            mv = mv.to_frame()
        #st.dataframe(fix_dataframe_types(mv), width="stretch")
        mv = mv.astype(str)
        st.dataframe(mv, width="stretch")

    # -------- TAB 3 --------
    with tab3:
        st.subheader("📈 Visualization")
        plot_df = df.copy()

        for col in plot_df.columns:
            try:
                plot_df[col] = pd.to_numeric(plot_df[col])
            except:
                pass

        numeric_cols, categorical_cols = get_columns(plot_df)
        chart_type = st.selectbox("Chart Type",["Histogram", "Boxplot", "Countplot", "Scatter"])

        col = None
        col2 = None

        if chart_type == "Scatter":
            if len(numeric_cols) >= 2:
                col = st.selectbox("Select X-axis", numeric_cols)
                col2 = st.selectbox(
                    "Select Y-axis",
                    [c for c in numeric_cols if c != col]
                )
        elif chart_type == "Countplot":
            if len(categorical_cols) > 0:
                col = st.selectbox("Select Column", categorical_cols)
        else:
            if len(numeric_cols) > 0:
                col = st.selectbox("Select Column", numeric_cols)

        if col is not None:
            fig = plot_graph(plot_df, col, chart_type, col2)
            st.pyplot(fig)

        # Heatmap
        st.subheader("🔥 Correlation Heatmap")

        if len(numeric_cols) > 1:
            threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.3)

            if len(numeric_cols) <= 2:
                max_cols = len(numeric_cols)
            else:
                max_cols = st.slider(
                    "Max Columns for Heatmap",
                    2,
                    len(numeric_cols),
                    min(10, len(numeric_cols))
                )

            st.pyplot(correlation_heatmap(df, threshold=threshold, max_cols=max_cols))
            
        # Correlation With Target
        st.subheader("📊 Correlation with Target")
        target_col = st.selectbox("Select Target for Correlation", df.columns)
        from sklearn.preprocessing import LabelEncoder
        temp_df = df.copy()
        # Encode target if categorical
        if temp_df[target_col].dtype == "object":
            temp_df[target_col] = LabelEncoder().fit_transform(temp_df[target_col])

        # Correlation
        if target_col in temp_df.select_dtypes(include='number').columns:
            corr_target = temp_df.corr(numeric_only=True)[target_col].sort_values(ascending=False)

            st.subheader("📊 Feature Correlation with Target")
            st.dataframe(corr_target.to_frame().astype(str), width="stretch")
        else:
            st.warning("⚠️ Target column not suitable for correlation")
    
    #-------------------Tab 4------------------------
    with tab4:
        st.subheader("🧠 AI Insights")
        if st.button("🧠 Generate AI Insights"):
            import pandas as pd
            summary = {
                "Shape" : df.shape,
                "Columns" : list(df.columns),
                "Missing Values" : df.isnull().sum().to_dict(),
                "Data Types" : df.dtypes.astype(str).to_dict(),
                "Describe" : df.describe().round(2).to_dict()
            }
            
            prompt = f"""
            You are a data analyst.
            Analyze this dataset and provide key insights, issues and recommendation.
            
            DATASET SUMMARY:
            
            {summary}
            
            Give:
            - Key patterns
            - Data Quality issues
            - Suggestions for ML
            """
            
            # Call LLM
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
            try:
                response = llm.invoke(prompt)
                st.write(str(response.content))
            except Exception as e:
                st.error(f"❌ LLM Error: {e}")
            
    # -------- TAB 5 (FIXED COMPARISON) --------
    with tab5:
        st.subheader("📊 Data Cleaning Steps Summary")

        if st.session_state.cleaned_df is None:
            st.warning("⚠️ Click 'Clean Data' first")
        else:
            st.write("Original:", st.session_state.base_df.shape)
            st.write("After Dropping the column:", st.session_state.original_df.shape)
            if st.session_state.cleaned_df is not None:
                st.write("After Cleaning:", st.session_state.cleaned_df.shape)

            with st.expander("Cleaning Report"):
                for r in st.session_state.cleaning_report:
                    st.write("👉", r)


    # -------- TAB 6  ML MODEL --------
    with tab6:
        st.title("🤖 Machine Learning Studio")

        # ------------ Step 0: Setup ---------------
        st.header("⚙️ Step 0: Setup")
        target = st.selectbox("🎯 Select Target Column", df.columns)
        task = st.radio("🧠 Select Task", ["Classification", "Regression"])
        model_choice = st.selectbox(
            "📦 Select Model",
            ["Random Forest", "Linear Regression", "Logistic Regression", "XGBoost"]
        )
        st.divider()

        # ------------ Hyperparameters ---------------
        if model_choice == "Linear Regression":
            st.info("ℹ️ Linear Regression does not require hyperparameter tuning.")
            
            
        st.subheader("🧪 Hyperparameter Options")
        param_inputs = {}

        if model_choice == "Random Forest":
            param_inputs = {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        elif model_choice == "Logistic Regression":
            param_inputs = {
                "C": [0.01, 0.1, 1, 10]
            }

        # ------------ Get Model Function ---------------
        

        # ------------ Step 1: Train Base Model ---------------
        st.header("🚀 Step 1: Train Base Model")

        if st.button("Train Base Model"):

            import numpy as np
            import pandas as pd

            X = df.drop(columns=[target]).copy()
            # 
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col])
                except:
                    pass
            y = df[target]
            
            st.session_state.original_columns = X.columns.tolist()
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            st.session_state.categorical_cols = cat_cols
            st.session_state.cat_values = {
            col: X[col].dropna().unique().tolist() for col in cat_cols
            }

            X, y, le = preprocess_data(X, y, model_choice)
            st.session_state.label_encoder = le
            st.session_state.model_columns = X.columns.tolist()
            
            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            from sklearn.linear_model import LinearRegression
            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.session_state.trained_model = model
                st.session_state.y_pred = preds
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.model_columns = X.columns
                st.session_state.scaler = None
                # Metrics
                from sklearn.metrics import mean_squared_error, r2_score

                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                st.metric("MSE", f"{mse:.2f}")
                st.metric("R²", f"{r2:.2f}")

                score = r2

                # Plot
                st.subheader("📉 Predictions vs Actual")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()

                ax.scatter(y_test, preds, alpha=0.6)

                min_val = min(min(y_test), min(preds))
                max_val = max(max(y_test), max(preds))

                ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")

                st.pyplot(fig)
                st.subheader("📋 Actual vs Predicted Values")
                comparison_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": preds
                })
                st.dataframe(comparison_df.head(10))

                # Save scores
                st.session_state.base_score = score
                st.session_state.final_score = score
                st.success(f"🎯 Final Model Score: {score:.4f}")
            else:
                
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)   # ✅ fit ONLY on train
                X_test = scaler.transform(X_test)         # ✅ transform test

                # Save scaler
                st.session_state.scaler = scaler

                # Save splits
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test

                # Train model
                model = get_model(model_choice, task)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            

                # Save model
                st.session_state.y_pred = preds
                st.session_state.trained_model = model

                # Score
                from sklearn.metrics import accuracy_score, r2_score
                score = accuracy_score(y_test, preds) if task == "Classification" else r2_score(y_test, preds)

                st.session_state.base_score = score

                st.success(f"✅ Base Model Score: {score:.4f}")

        st.divider()
        if model_choice != "Linear Regression" and "trained_model" in st.session_state:
                    
            # ------------ Step 2: Compare Tuning ---------------
            st.header("⚙️ Step 2: Compare Tuning Methods")

            if st.button("Compare Grid Vs Random"):
                if "X_train" not in st.session_state:
                    st.warning("Train Model First!")
                    st.stop()

                from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                base_model = get_model(model_choice, task)
                if not param_inputs:
                    st.warning("⚠️ No hyperparameters available for this model.")
                    st.stop()

                total_combinations = 1
                for v in param_inputs.values():
                    total_combinations *= len(v)
                
                n_iter = min(5, total_combinations)
                grid = GridSearchCV(base_model, param_inputs, cv = 3, n_jobs = -1)
                random = RandomizedSearchCV(
                    base_model,
                    param_inputs,
                    n_iter = n_iter,
                    cv = 3,
                    n_jobs = -1,
                    random_state = 42 
                )

                grid.fit(X_train, y_train)
                random.fit(X_train, y_train)

                st.session_state.grid_score = grid.best_score_
                st.session_state.random_score = random.best_score_
                st.session_state.grid_params = grid.best_params_
                st.session_state.random_params = random.best_params_

                st.success("✅ Comparison Done!")
                st.write("Grid Score:", grid.best_score_)
                st.write("Random Score:", random.best_score_)

        st.divider()

        # ------------ Step 3: Choose Best ---------------
        if "grid_score" in st.session_state:
            st.header("🏆 Step 3: Choose Best Method")

            method = st.radio("Select Method", ["GridSearchCV", "RandomizedSearchCV"])

            best_params = (
                st.session_state.grid_params
                if method == "GridSearchCV"
                else st.session_state.random_params
            )

            st.session_state.best_params = best_params
            st.json(best_params)

        # ------------ Step 4: Final Train ---------------
        if "best_params" in st.session_state:
            st.header("🚀 Step 4: Train Final Model")

            import matplotlib.pyplot as plt
            import pickle
            from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

            # 🔒 Guard (prevents crash if user skips steps)
            required_keys = ["X_train", "X_test", "y_train", "y_test"]
            for key in required_keys:
                if key not in st.session_state:
                    st.error(f"⚠️ Missing {key}. Please complete previous steps.")
                    st.stop()

            user_params = {}

            # ✅ UNIQUE KEYS (VERY IMPORTANT)
            for i, (k, v) in enumerate(st.session_state.best_params.items()):
                if isinstance(v, int):
                    user_params[k] = int(
                        st.number_input(k, value=v, step=1, key=f"num_{k}_{i}")
                    )
                elif isinstance(v, float):
                    user_params[k] = float(
                        st.number_input(k, value=v, key=f"float_{k}_{i}")
                    )
                else:
                    val = st.text_input(k, value=str(v), key=f"text_{k}_{i}")
                    user_params[k] = None if val == "None" else val

            # 🚀 Train Button
            if st.button("Train Final Model"):

                # ✅ Train model
                model = get_model(model_choice, task, user_params)
                model.fit(st.session_state.X_train, st.session_state.y_train)

                preds = model.predict(st.session_state.X_test)
                y_test = st.session_state.y_test

                # ✅ Save
                st.session_state.trained_model = model
                st.session_state.y_pred = preds

                # ---------------- METRICS ----------------
                if task == "Classification":
                    score = accuracy_score(y_test, preds)
                    st.metric("Accuracy", f"{score:.2f}")

                else:
                    mse = mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)

                    st.metric("MSE", f"{mse:.2f}")
                    st.metric("R²", f"{r2:.2f}")

                    score = r2

                    # 📉 Plot
                    st.subheader("📉 Predictions vs Actual")
                    fig, ax = plt.subplots()

                    ax.scatter(y_test, preds, alpha=0.6)

                    min_val = min(min(y_test), min(preds))
                    max_val = max(max(y_test), max(preds))

                    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")

                    st.pyplot(fig)

                # ✅ Save score
                st.session_state.final_score = score

                # ---------------- RESULT ----------------
                st.success(f"🎯 Final Model Score: {score:.4f}")

                # ---------------- DOWNLOAD ----------------
                model_package = {
                    "model": model,
                    "scaler": st.session_state.get("scaler"),
                    "label_encoder": st.session_state.get("label_encoder"),
                    "columns": st.session_state.get("model_columns"),
                    "categorical_cols": st.session_state.get("categorical_cols"),
                    "cat_values": st.session_state.get("cat_values")
                }

                st.download_button(
                    "📥 Download Model Package",
                    data=pickle.dumps(model_package),
                    file_name="model_package.pkl"
                )
        
        # ------------ Performance Comparison ---------------
        st.header("📊 Performance Comparison")

        data = []
        labels = []

        if "base_score" in st.session_state:
            data.append(st.session_state.base_score)
            labels.append("Base")

        if "grid_score" in st.session_state:
            data.append(st.session_state.grid_score)
            labels.append("Grid")

        if "random_score" in st.session_state:
            data.append(st.session_state.random_score)
            labels.append("Random")

        if "final_score" in st.session_state:
            data.append(st.session_state.final_score)
            labels.append("Final")

        if data:
            import pandas as pd
            df_plot = pd.DataFrame({"Stage": labels, "Score": data})
            st.bar_chart(df_plot.set_index("Stage"))
            
        
        #------------------Step 5: Detailed Result-------------
        if "trained_model" in st.session_state and "y_pred" in st.session_state:
            st.header("📊 Detailed Model Insights")
            model = st.session_state.trained_model
            y_pred = st.session_state.y_pred
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            #--------------Feature Importance-------------
            with st.expander("📊 Feature Importance", expanded=True):
                importance = None

                # Tree models
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_

                # Linear models
                elif hasattr(model, "coef_"):
                    importance = model.coef_.flatten()

                # Fallback (ALL models)
                else:
                    from sklearn.inspection import permutation_importance
                    result = permutation_importance(
                        model,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        n_repeats=5,
                        random_state=42
                    )
                    importance = result.importances_mean

                # Display
                if importance is not None:
                    feature_df = pd.DataFrame({
                        "Feature": st.session_state.model_columns,
                        "Importance": importance
                    }).sort_values(by="Importance", ascending=False)

                    st.dataframe(feature_df.astype(str))

                    top10 = feature_df.head(10)
                    fig, ax = plt.subplots()
                    ax.barh(top10["Feature"], top10["Importance"])
                    ax.invert_yaxis()
                    st.pyplot(fig)
            
            #----------------------------Metrics-----------------
            from sklearn.metrics import(
                accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
            ) 
            with st.expander("📈 Metrics", expanded=True):
                if task == "Classification":
                    acc = accuracy_score(st.session_state.y_test, y_pred)
                    st.metric("Accuracy", f"{acc :.4f}")  
                else:
                    mse = mean_squared_error(st.session_state.y_test, y_pred)
                    r2 = r2_score(st.session_state.y_test, y_pred)
                    st.metric("MSE", f"{mse :.4f}") 
                    st.metric("R2 Score", f"{r2 :.4f}")
                    
            #---------------------Classification Report---------
            if task == "Classification":
                with st.expander("📄 Classification Report"):
                    report = classification_report(
                        st.session_state.y_test,
                        y_pred,
                        output_dict = True
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.astype(str))
                    
            #------------------Confusion Matrix------------
            if task == "Classification":
                with st.expander("🧩 Confusion Matrix"):
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    fig, ax = plt.subplots()
                    try:
                        import seaborn as sns 
                        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                    except:
                        ax.imshow(cm)
                        for i in range(len(cm)):
                            for j in range(len(cm)):
                                ax.text(j, i, cm[i, j], ha="center", va="center")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    
            #------------------Sample Predictions------------
            with st.expander("🔍 Sample Predictions of first 10 dataset"):
                sample_df = pd.DataFrame({
                    "Actual" : st.session_state.y_test[:10],
                    "Predicted" : y_pred[:10]
                })
                st.dataframe(sample_df.astype(str))
    #------------------------TAB 7---------------------------
    with tab7:
        st.subheader("🤖 AI Data Scientist Assistant")
        if st.button("📊 Suggest Visualizations"):
            summary = {
                "Columns" : list(df.columns),
                "Data Types" : df.dtypes.astype(str).to_dict()
            }    
            prompt = f"""
            You are a data analyst.
            
            Dataset:
            Columns: {summary['Columns']}
            Data Types: {summary['Data Types']}
            
            Suggest Best Visualization
            
            Format:
            Column --> Chart Type
            
            """
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
            
            response = llm.invoke(prompt)
            st.subheader("📊 AI Suggested Visualizations")
            st.write(str(response.content))
            
        st.divider()
        
       # -------------- MODEL SELECTION---------
        target_ai = st.selectbox("🎯 Select Target for AI Model", df.columns)
        
        if st.button("🤖 Suggest Best Model"):
            summary = {
                "Target" : target_ai,
                "Data Types" : df.dtypes.astype(str).to_dict(),
                "Sample" : df.head(5).to_string()
            }
            
            prompt = f"""
            You are a Machine Learning Expert.
            
            Dataset info:
            Target : {summary['Target']}
            Data Types : {summary['Data Types']}
            Sample :
            {summary['Sample']}
            Explain:
            1. Problem type (Classification / Regression)
            2. Best Model
            3. Reason 
            
            Highlight important thing
            Keep it short.
            """
            
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
            response = llm.invoke(prompt)
            st.subheader("🤖 AI Model Recommendation")
            st.write(str(response.content))
            
            # Auto Model Selection
            st.subheader("🤖 AI Model Recommendation")
            import pandas as pd
            X = df.drop(columns = [target_ai])
            y = df[target_ai]
            
            text = response.content.lower()
            # Classification
            if "classification" in text:
                task = "Classification"
                if "xgboost" in text:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators = 100,
                        max_depth = 5,
                        learning_rate = 0.1,
                        eval_metric = 'logloss'
                    )
                    model_choice = "XGBoost"
                    model_name = "XGBoost Classifier"
                elif "random forest" in text:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier()
                    model_choice = "Random Forest"
                    model_name = "Random Forest Classifier"
                    
                else:
                    from sklearn .linear_model import LogisticRegression
                    model = LogisticRegression(max_iter = 5000, solver = "lbfgs", n_jobs = -1)
                    model_choice = "Logistic Regression"
                    model_name = "Logistic Regression"
                    
            # Regression
            else:
                task = "Regression"
                if "xgboost" in text:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        n_estimators = 100,
                        max_depth = 5,
                        learning_rate = 0.1
                    )
                    model_choice = "XGBoost"
                    model_name = "XGBoost Regressor"
                elif "random forest" in text:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor()
                    model_choice = "Random Forest"
                    model_name = "Random Forest Regressor"
                else:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model_choice = "Linear Regression"
                    model_name = "Linear Regression"
                    
            X, y, _ = preprocess_data(X, y, model_choice)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
                    
            st.success(f"🚀 Using Model: {model_name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Metrics 
            if "classification" in response.content.lower():
                from sklearn.metrics import accuracy_score
                st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                st.metric("MSE", f"{mean_squared_error(y_test, preds):.2f}")
                st.metric("R^2", f"{r2_score(y_test, preds):.2f}")
         
         
    #--------------------------- TAB 8-----------------------------------
    with tab8:
        st.subheader("⚔️ Model Comparison")    
        target = st.selectbox("Select Target", df.columns, key = "compare_target")
        task = st.selectbox("Task Type", ["Classification", "Regression"], key = "compare_task")
        
        if st.button("🚀 Run Comparison"):
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score  
            
            X = df.drop(columns = [target])
            y = df[target]
            import numpy as np
            import pandas as pd 
            X = pd.get_dummies(X, drop_first = True)
            X = X.replace([np.inf, -np.inf], 0).fillna(0)
            if y.dtype == "object":
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)
            # Split the data into train, test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.2, random_state = 42
            )
            results = []
            #--------------MODELS-----------------------
            if task == "Classification":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from xgboost import XGBClassifier
                
                models = {
                    "Random Forest" : RandomForestClassifier(),
                    "Logistic Regression" : LogisticRegression(max_iter = 5000, solver = "lbfgs", n_jobs = -1),
                    "XGBoost" : XGBClassifier(eval_metric = 'logloss')
                }
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    
                    results.append({
                        "Model" : name,
                        "Accuracy" : acc
                    })
            else:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.linear_model import LinearRegression
                from xgboost import XGBRegressor
                
                models = {
                    "Linear Regression" : LinearRegression(),
                    "Random Forest" : RandomForestRegressor(),
                    "XGBoost" : XGBRegressor()
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    
                    results.append({
                        "Model" : name,
                        "MSE" : mse,
                        "R2 Score" : r2
                    })
                    
            #-----------------RESULTS-----------------
            results_df = pd.DataFrame(results)
            st.subheader("📊 Comparison Results")
            st.dataframe(results_df.astype(str))
            
            # Highlight the best Model
            if task == "Classification":
                best = results_df.sort_values(by = "Accuracy", ascending = False).iloc[0]
                st.success(f"🏆 Best Model: {best['Model']} (Accuracy: {best['Accuracy']:.2f})")
            else:
                best = results_df.sort_values(by = "R2 Score", ascending = False).iloc[0]
                st.success(f"🏆 Best Model: {best['Model']} (R²: {best['R2 Score']:.2f})")
                st.bar_chart(results_df.set_index("Model"))
                     
   #---------------------PREDICTION----------------------   
    with tab9:
        st.header("🔮 Prediction")
        if "trained_model" not in st.session_state:
            st.warning("⚠️ Please train the model first")
        else:
            import pandas as pd
            import numpy as np
            model = st.session_state.trained_model
            cat_cols = st.session_state.categorical_cols
            cat_values = st.session_state.cat_values
            model_cols = st.session_state.model_columns
            st.subheader("📥 Enter Input Data")
            user_input = {}
            
            for col in st.session_state.original_columns:
                if col in cat_cols:
                    user_input[col] = st.selectbox(col, cat_values[col])
                else:
                    user_input[col] = st.number_input(col, value=0.0)
            
            input_df = pd.DataFrame([user_input])
            # 🔥 Convert to dummy (same as training)
            input_df = pd.get_dummies(input_df)

            # 🔥 Align columns with training
            input_df = input_df.reindex(columns=model_cols, fill_value=0)

            if st.session_state.scaler is not None:
                input_data = st.session_state.scaler.transform(input_df)
            else:
                input_data = input_df
            
            if st.button("predict"):
                
                pred = model.predict(input_data)[0]

                # 🔥 Decode label
                if st.session_state.get("label_encoder") is not None:
                    pred = st.session_state.label_encoder.inverse_transform([pred])[0]

                st.markdown("### 🎯 Prediction Result")
                st.success(f"Prediction: {pred}")
                    
else:
    st.write("👈 Upload CSV")