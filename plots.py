import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np   

def plot_graph(df, column, chart_type, col2=None):
    fig, ax = plt.subplots()

    if chart_type == "Histogram":
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    elif chart_type == "Boxplot":
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")  
        ax.set_xlabel(column)

    elif chart_type == "Countplot":
        sns.countplot(x=df[column], ax=ax)
        ax.set_title(f"Countplot of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)

    elif chart_type == "Scatter":
        if col2 is not None:
            sns.scatterplot(x=df[column], y=df[col2], ax=ax)
            ax.set_title(f"{column} vs {col2}")
            ax.set_xlabel(column)
            ax.set_ylabel(col2)   
        else:
            ax.text(0.5, 0.5, "Select two columns for scatter",
                    ha='center', va='center')

    plt.tight_layout()  
    return fig

def correlation_heatmap(df, threshold=0.5, max_cols=10):
    numeric_df = df.select_dtypes(include='number')

    # Limit columns
    numeric_df = numeric_df.iloc[:, :max_cols]

    corr = numeric_df.corr()

    # Create mask for lower correlations
    mask_low = np.abs(corr) < threshold

    # Mask upper triangle
    mask_upper = np.triu(np.ones_like(corr, dtype=bool))

    # Combine masks
    final_mask = mask_upper | mask_low

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        mask=final_mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title(f"Correlation Heatmap (>|{threshold}|)")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    return fig
