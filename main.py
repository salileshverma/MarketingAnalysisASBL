import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)
warnings.filterwarnings("ignore")




def safe_divide(numerator, denominator):
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def save_plot(filename):
    """Save current plot to outputs/plots/ and close."""
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig(f"outputs/plots/{filename}", bbox_inches="tight")
    plt.close()


# ---------- Section 1 ---------- #

def load_marketing_data():
    path = "data/marketing_dataset.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def show_basic_info(df):
    print_section("SECTION 1: Basic Info")
    print("First 10 rows:\n", df.head(10))
    print("\nShape:", df.shape)
    print("\nSummary statistics:\n", df.describe())


def convert_and_sort_dates(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date")


def list_unique_values(df):
    print("\nUnique Platforms:", df["Platform"].unique())
    print("Unique Projects:", df["Project"].unique())
    print("Unique URLs:", df["URL"].unique())
    print("Unique Adsets:", df["Adset"].unique())


def filter_google_high_visitors(df):
    filtered = df[(df["Platform"] == "Google") & (df["Visitors"] > 10000)]
    print("\nFiltered rows (Google + Visitors > 10k):\n", filtered.head())
    return filtered


def add_cpl_and_top_adsets(df):
    df["CPL"] = safe_divide(df["Spend"], df["Leads"])
    adset_summary = df.groupby("Adset").agg(
        Spend_sum=("Spend", "sum"),
        Leads_sum=("Leads", "sum")
    ).reset_index()
    adset_summary["CPL"] = safe_divide(adset_summary["Spend_sum"], adset_summary["Leads_sum"])
    print("\nTop 5 expensive adsets:\n", adset_summary.sort_values("CPL", ascending=False).head(5))
    return df


# ---------- Section 2 ---------- #

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
    print("\nProject with highest SiteVisits:\n", proj_visits.sort_values("SiteVisits", ascending=False).head(1))


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
    print("\nAnomalies:\n", anomalies)


# ---------- Section 3 ---------- #

def best_adset_per_platform(df):
    grouped = df.groupby(["Platform", "Adset"])["Closure"].sum().reset_index()
    idx = grouped.groupby("Platform")["Closure"].idxmax()
    print_section("SECTION 3: Best Adset per Platform")
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
    os.makedirs("outputs", exist_ok=True)
    dashboard.to_csv("outputs/daily_dashboard.csv", index=False)
    print("\nSaved daily_dashboard.csv in outputs/")
    return dashboard


def funnel_summary(df):
    funnel = df.groupby("Project").agg(
        Visitors=("Visitors", "sum"),
        Leads=("Leads", "sum"),
        SiteVisits=("SiteVisits", "sum"),
        Closure=("Closure", "sum")
    ).reset_index()
    print("\nFunnel Summary:\n", funnel.head())


# ---------- Section 4 ---------- #

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
    plt.title("Daily Visitors (Top 3 Platforms)")
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
    main()
