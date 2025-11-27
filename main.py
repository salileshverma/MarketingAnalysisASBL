import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ CONFIG ------------------ #
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
warnings.filterwarnings("ignore")

DATA_PATH = "data/marketing_dataset.csv"
OUTPUT_DIR = "outputs"
PLOT_DIR = f"{OUTPUT_DIR}/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ------------------ HELPER FUNCTIONS ------------------ #

def safe_divide(numerator, denominator):
    """Safely divide values avoiding division by zero."""
    if hasattr(denominator, "replace"):
        denominator = denominator.replace(0, np.nan)
    denominator = np.where(denominator == 0, np.nan, denominator)
    return numerator / denominator


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def save_plot(filename):
    """Save current plot and close."""
    plt.savefig(f"{PLOT_DIR}/{filename}", bbox_inches="tight")
    plt.close()


# ------------------ SECTION 1 ------------------ #

def load_marketing_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"ERROR: Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def show_basic_info(df):
    print_section("SECTION 1: Basic Information")
    print("First 10 rows:\n", df.head(10))
    print("\nShape (Rows, Columns):", df.shape)
    print("\nSummary Statistics:\n", df.describe())


def convert_and_sort_dates(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def list_unique_values(df):
    print("\nUnique Platforms:", df["Platform"].unique())
    print("Unique Projects:", df["Project"].unique())
    print("Unique URLs:", df["URL"].unique())
    print("Unique Adsets:", df["Adset"].unique())


def filter_google_high_visitors(df):
    filtered = df[(df["Platform"] == "Google") & (df["Visitors"] > 10000)]
    print("\nGoogle rows with > 10k visitors:\n", filtered.head())
    return filtered


def add_cpl_and_top_adsets(df):
    df["CPL"] = safe_divide(df["Spend"], df["Leads"])
    summary = df.groupby("Adset").agg(
        Spend=("Spend", "sum"),
        Leads=("Leads", "sum")
    ).reset_index()
    summary["CPL"] = safe_divide(summary["Spend"], summary["Leads"])
    print("\nTop 5 Most Expensive Adsets (CPL):\n", summary.sort_values("CPL", ascending=False).head())
    return df


# ------------------ SECTION 2 ------------------ #

def platform_group_metrics(df):
    grouped = df.groupby("Platform").agg(
        Total_Spend=("Spend", "sum"),
        Total_Visitors=("Visitors", "sum"),
        Total_Leads=("Leads", "sum"),
        Total_Closure=("Closure", "sum")
    ).reset_index()

    grouped["Avg_CPL"] = safe_divide(grouped["Total_Spend"], grouped["Total_Leads"])

    print_section("SECTION 2: Platform Metrics")
    print(grouped)
    return grouped


def project_highest_sitevisits(df):
    proj_visits = df.groupby("Project")["SiteVisits"].sum().reset_index()
    print("\nProject with Highest SiteVisits:\n",
          proj_visits.sort_values("SiteVisits", ascending=False).head(1))


def plot_daily_spend_trend(df):
    daily = df.groupby(["Date", "Platform"])["Spend"].sum().reset_index()
    sns.lineplot(data=daily, x="Date", y="Spend", hue="Platform", marker="o")
    plt.title("Daily Spend Trend by Platform")
    save_plot("daily_spend_trend.png")


def add_funnel_metrics(df):
    df["Lead_Conversion"] = safe_divide(df["Leads"], df["Visitors"])
    df["SiteVisit_Conversion"] = safe_divide(df["SiteVisits"], df["Leads"])
    df["Closure_Conversion"] = safe_divide(df["Closure"], df["SiteVisits"])
    return df


def detect_anomalies(df):
    daily = df.groupby("Date").agg(Visitors=("Visitors", "sum"), Leads=("Leads", "sum")).reset_index()

    for col in ["Visitors", "Leads"]:
        mean, std = daily[col].mean(), daily[col].std()
        daily[f"{col}_zscore"] = (daily[col] - mean) / (std if std else np.nan)

    anomalies = daily[(daily["Visitors_zscore"].abs() > 2) | (daily["Leads_zscore"].abs() > 2)]
    print("\nDetected Anomalies:\n", anomalies)


# ------------------ SECTION 3 ------------------ #

def best_adset_per_platform(df):
    grouped = df.groupby(["Platform", "Adset"])["Closure"].sum().reset_index()
    idx = grouped.groupby("Platform")["Closure"].idxmax()

    print_section("Best Adset Per Platform (Highest Closure)")
    print(grouped.loc[idx])


def correlation_heatmap(df):
    corr = df[["Spend", "Visitors", "Leads", "SiteVisits", "Closure"]].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")


def build_daily_dashboard(df):
    dashboard = df.groupby("Date").agg(
        Total_Spend=("Spend", "sum"),
        Total_Visitors=("Visitors", "sum"),
        Total_Leads=("Leads", "sum"),
        Total_SiteVisits=("SiteVisits", "sum"),
        Total_Closure=("Closure", "sum")
    ).reset_index()

    dashboard.to_csv(f"{OUTPUT_DIR}/daily_dashboard.csv", index=False)
    print("\nSaved daily_dashboard.csv to /outputs/")
    return dashboard


def funnel_summary(df):
    summary = df.groupby("Project").agg(
        Visitors=("Visitors", "sum"),
        Leads=("Leads", "sum"),
        SiteVisits=("SiteVisits", "sum"),
        Closure=("Closure", "sum")
    )
    print("\nProject Funnel Summary:\n", summary)


# ------------------ SECTION 4 ------------------ #

def plot_spend_vs_leads(df):
    sns.scatterplot(data=df, x="Spend", y="Leads", hue="Platform")
    plt.title("Spend vs Leads")
    save_plot("spend_vs_leads.png")


def plot_avg_closure(df):
    avg = df.groupby("Platform")["Closure"].mean().reset_index()
    sns.barplot(data=avg, x="Platform", y="Closure")
    plt.title("Average Closure per Platform")
    save_plot("avg_closure.png")


def plot_top3_visitors(df):
    top3 = df.groupby("Platform")["Visitors"].sum().nlargest(3).index
    daily = df[df["Platform"].isin(top3)].groupby(["Date", "Platform"])["Visitors"].sum().reset_index()
    sns.lineplot(data=daily, x="Date", y="Visitors", hue="Platform", marker="o")
    plt.title("Top 3 Platforms - Daily Visitors")
    save_plot("top3_visitors.png")


def platform_dashboard(df):
    daily = df.groupby(["Date", "Platform"]).agg(
        Spend=("Spend", "sum"),
        Visitors=("Visitors", "sum"),
        Leads=("Leads", "sum"),
        Closure=("Closure", "sum")
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = ["Spend", "Visitors", "Leads", "Closure"]
    titles = ["Spend Trend", "Visitors Trend", "Leads Trend", "Closure Trend"]

    for ax, col, title in zip(axes.flatten(), metrics, titles):
        sns.lineplot(data=daily, x="Date", y=col, hue="Platform", ax=ax, marker="o")
        ax.set_title(title)

    plt.tight_layout()
    save_plot("platform_dashboard.png")


# ------------------ MAIN RUNNER ------------------ #

def main():
    df = load_marketing_data()

    show_basic_info(df)
    df = convert_and_sort_dates(df)
    list_unique_values(df)
    filter_google_high_visitors(df)
    df = add_cpl_and_top_adsets(df)

    platform_group_metrics(df)
    project_highest_sitevisits(df)
    plot_daily_spend_trend(df)

    df = add_funnel_metrics(df)
    detect_anomalies(df)

    best_adset_per_platform(df)
    correlation_heatmap(df)
    build_daily_dashboard(df)
    funnel_summary(df)

    plot_spend_vs_leads(df)
    plot_avg_closure(df)
    plot_top3_visitors(df)
    platform_dashboard(df)


if __name__ == "__main__":
    print("\n📊 Running Marketing Data Analysis...\n")
    main()
    print("\n✅ Analysis Completed. Check 'outputs/' folder for results.\n")
