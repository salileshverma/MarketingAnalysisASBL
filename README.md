📊 Digital Marketing Data Analysis Project
📌 Overview

This project analyzes real-world marketing performance data using Python, Pandas, NumPy, and Matplotlib.
The goal is to clean, transform, explore, visualize, and extract insights from a dataset containing digital marketing metrics such as spend, leads, visitors, closures, platform performance, and campaign efficiency.

This assignment demonstrates:

Data cleaning

Exploratory data analysis (EDA)

Feature engineering and funnel metrics

Statistical anomaly detection

Visualization of campaign performance

Dashboard and reporting output

The project outputs multiple charts and summary tables, including a daily performance dashboard CSV.

📁 Project Structure

📂 Marketing-Analysis-Project
│
├── marketing_dataset.csv         
├── analysis_report.csv          
├── daily_dashboard.csv           
├── platform_performance.csv      
│
├── plots/                        -> auto-generated graphs & dashboards
│   ├── spend_vs_leads_scatter.png
│   ├── avg_closure_by_platform.png
│   ├── daily_visitors_top3_platforms.png
│   ├── heatmap_correlations.png
│   ├── platform_dashboard_Facebook.png
│   ├── (other platform dashboards)
│
├── main.py                       -> main executable script
├── requirements.txt              -> dependencies
└── README.md                     -> project documentation

🚀 How to Run the Project
1️⃣ Install Dependencies

Open terminal or PyCharm terminal and run:
pip install -r requirements.txt
2️⃣ Run the Script
python main.py

🏁 Output Files Generated
| File                       | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `analysis_report.csv`      | Insight summary including conversions & CPL |
| `daily_dashboard.csv`      | Aggregated per-day metrics                  |
| `platform_performance.csv` | KPI summary grouped by platform             |
| PNG charts                 | Visual insights                             |

