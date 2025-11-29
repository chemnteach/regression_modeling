
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from io import StringIO
import os

# Try to load from local file first
local_files = ['vix_data.csv', 'VIXCLS.csv', '^VIX.csv']
local_file = None

for filename in local_files:
    if os.path.exists(filename):
        local_file = filename
        break

if local_file:
    print(f"📁 Loading VIX data from local file: {local_file}")
    try:
        vix = pd.read_csv(local_file)
        
        # Handle FRED format (DATE, VIXCLS columns)
        if 'VIXCLS' in vix.columns:
            vix.columns = ['Date', 'Close']
        
        if 'Date' in vix.columns:
            vix['Date'] = pd.to_datetime(vix['Date'])
            vix.set_index('Date', inplace=True)
        else:
            vix.index = pd.to_datetime(vix.index)
        
        # Convert Close to numeric and drop NaN
        vix['Close'] = pd.to_numeric(vix['Close'], errors='coerce')
        vix = vix.dropna()
        
        # Filter to last 5 years
        five_years_ago = datetime.now() - timedelta(days=365*5)
        vix = vix[vix.index >= five_years_ago]
        
        print(f"✅ Successfully loaded {len(vix)} records from local file")
    except Exception as e:
        print(f"❌ Failed to load local file: {e}")
        exit()
else:
    # Download VIX data from FRED (Federal Reserve Economic Data)
    print("📥 Downloading VIX data from FRED (Federal Reserve)...")
    
    # FRED VIX data URL (free, no subscription needed)
    # VIXCLS = CBOE Volatility Index: VIX
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    
    try:
        session = requests.Session()
        session.trust_env = False  # Ignore system proxy settings
        response = session.get(url, verify=False, timeout=30)
        response.raise_for_status()
        
        # Read CSV from the response text
        vix = pd.read_csv(StringIO(response.text))
        vix.columns = ['Date', 'Close']
        vix['Date'] = pd.to_datetime(vix['Date'])
        vix.set_index('Date', inplace=True)
        
        # Filter to last 5 years
        five_years_ago = datetime.now() - timedelta(days=365*5)
        vix = vix[vix.index >= five_years_ago]
        
        # Remove any missing values (represented as '.' in FRED data)
        vix['Close'] = pd.to_numeric(vix['Close'], errors='coerce')
        vix = vix.dropna()
        
        print(f"✅ Successfully downloaded {len(vix)} records from FRED")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n" + "="*60)
        print("ALTERNATIVE DOWNLOAD:")
        print("="*60)
        print("1. Visit: https://fred.stlouisfed.org/series/VIXCLS")
        print("2. Click 'Download' button (top right)")
        print("3. Choose CSV format")
        print("4. Save file to this folder and rename to: vix_data.csv")
        print(f"   {os.getcwd()}")
        print("5. Run this script again")
        print("="*60)
        exit()

# Check if data was downloaded
if vix.empty:
    print("Error: No data downloaded.")
    exit()

print(f"Downloaded {len(vix)} records")
print(f"Columns: {vix.columns.tolist()}")

# Handle MultiIndex columns if present
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Prepare the DataFrame
vix_monthly = vix[['Close']].copy()
vix_monthly = vix_monthly.dropna()  # Remove any NaN values
vix_monthly['MonthName'] = vix_monthly.index.month_name()
vix_monthly['Month'] = vix_monthly.index.month
vix_monthly['Year'] = vix_monthly.index.year

# Sort months in calendar order
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
vix_monthly['MonthName'] = pd.Categorical(
    vix_monthly['MonthName'],
    categories=month_order,
    ordered=True
)

# Create boxplot with yearly monthly averages
fig, ax = plt.subplots(figsize=(14, 7))

# Create a colormap for years
years = sorted(vix_monthly['Year'].unique())
colors = plt.cm.tab10(range(len(years)))

# Plot boxplot showing overall distribution
sns.boxplot(x='MonthName', y='Close', data=vix_monthly, ax=ax, 
            color='lightgray', width=0.6, fliersize=0)

# Calculate and plot monthly average for each year
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

for idx, year in enumerate(years):
    year_data = vix_monthly[vix_monthly['Year'] == year].copy()
    yearly_monthly_avg = year_data.groupby('MonthName')['Close'].mean().reindex(month_order)
    
    ax.scatter(range(len(yearly_monthly_avg)), yearly_monthly_avg.values, 
               s=150, label=str(year), color=colors[idx], alpha=0.8, 
               edgecolors='black', linewidth=1.5, zorder=3)

ax.set_title('📊 VIX Distribution by Month with Yearly Monthly Averages', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('VIX Close', fontsize=12)
plt.xticks(rotation=45)
ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
