from utils import *
import seaborn as sns
from scipy.stats import pearsonr

# plot the price data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plot the raw data
df, ax = compare(   
    "data/daily_9mm.csv",
    "data/daily_223.csv",
    "data/daily_x39.csv",
)
ax.legend()
ax.set_ylabel("Cartridge Price ($)")
fig = ax.get_figure()
fig.autofmt_xdate()
fig.savefig(path_adapt("raw_data.png"))

df.corr().to_markdown(path_adapt(f"caliber_correlations.md"))

# plot normalized values
df, ax = compare(
    "data/daily_9mm.csv",
    "data/daily_223.csv",
    "data/daily_x39.csv",
    normalize=True
)
ax.legend()
ax.set_ylabel("Cartridge Price (Normalized)")
fig = ax.get_figure()
fig.autofmt_xdate()
fig.savefig(path_adapt("norm_data.png"))

# read in data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nine = import_proph_data("./data/daily_9mm.csv")
r223 = import_proph_data("./data/daily_223.csv")
x762 = import_proph_data("./data/daily_x39.csv")

cci = import_CCI_data("data/CCI_US.csv")
oil = import_stock_data("data/CL=F.csv", "Close")
gas = import_stock_data("data/NG=F.csv", "Close")
cpr = import_stock_data("data/HG=F.csv", "Close")
ctn = import_stock_data("data/CT=F.csv", "Close")

dfs = {
    "9mm": nine, 
    "223": r223, 
    "762x39": x762, 
    "cci": cci, 
    "oil": oil, 
    "gas": gas, 
    "copper": cpr, 
    "cotton": ctn
    }

master = pd.DataFrame()
for n, df in dfs.items():
    df = df.rename(columns={df.columns[-1]: n})
    if len(master) == 0:
        master = df
    else:
        master = pd.merge(master, df, how="left")

# gather basic statistics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
master.set_index("ds", inplace=True)
master.corr().to_markdown(path_adapt("all_correlations.md"))
ax = master.pct_change(1).iloc[-20:].plot()
fig = ax.get_figure()
fig.savefig(path_adapt("raw_daily_returns.png"))

# check what type of seasonality each series is ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get log of day-to-day changes
rel_mags = master[1:] / master[:-1].values
log_prices = rel_mags.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
# log_prices = log_prices.add_suffix("_log")

# histograms of caliber price changes
bins = 20

fig, axs = plt.subplots(2,3)
master.pct_change(1)[["9mm", "223", "762x39"]].hist(ax=axs.flat[:3], bins=bins)
log_prices[["9mm", "223", "762x39"]].hist(ax=axs.flat[3:], bins=bins, color="tab:orange")
axs[0,0].set_ylabel("daily_returns", size="large")
axs[1,0].set_ylabel("log(daily_growth)", size="large")
fig.tight_layout()
fig.savefig(path_adapt("histograms_calibers.png"))

# histograms of cci
fig, axs = plt.subplots(2,1)
master.pct_change(1)[["cci",]].drop_duplicates().hist(ax=axs.flat[:1], bins=bins)
log_prices[["cci",]].drop_duplicates().hist(ax=axs.flat[1:], bins=bins, color="tab:orange")
axs[0].set_ylabel("monthly_returns", size="large")
axs[1].set_ylabel("log(monthly_growth)", size="large")
fig.tight_layout()
fig.savefig(path_adapt("histograms_cci.png"))

# histograms of futures prices
fig, axs = plt.subplots(2,4)
master.pct_change(1)[["oil", "gas", "copper", "cotton"]].hist(ax=axs.flat[:4], bins=bins)
log_prices[["oil", "gas", "copper", "cotton"]].hist(ax=axs.flat[4:], bins=bins, color="tab:orange")
axs[0,0].set_ylabel("daily_returns", size="large")
axs[1,0].set_ylabel("log(daily_growth)", size="large")
fig.tight_layout()
fig.savefig(path_adapt("histograms_futures.png"))

# raw scatter plots
fig, axs = plt.subplots(3,5)
for j, y in enumerate(["9mm", "223", "762x39"]):
    for i, x in enumerate(["cci", "oil", "gas", "copper", "cotton"]):
        master.plot.scatter(x, y, ax=axs[j, i])
fig.set_figheight(10)
fig.set_figwidth(15)
fig.tight_layout()
fig.savefig(path_adapt("scatters.png"))

# scatter plots of daily returns (excluding cci)
fig, axs = plt.subplots(3,4)
for j, y in enumerate(["9mm", "223", "762x39"]):
    for i, x in enumerate(["oil", "gas", "copper", "cotton"]):
        master.pct_change(1).plot.scatter(x, y, ax=axs[j, i])
fig.set_figheight(10)
fig.set_figwidth(15)
fig.tight_layout()
fig.savefig(path_adapt("daily_returns.png"))

# define function for annotating correlation coefficients
# https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot
def corr(x, y, ax=None, **kws):
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'corr coeff = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

# scatter plots of cartridge daily prices
pair_plot = sns.pairplot(
    master[["9mm", "223", "762x39"]].rename(columns=lambda x: x + " price ($)"),
    corner=True
    )
pair_plot.map_lower(corr)
fig = pair_plot.fig
fig.savefig(path_adapt("scatter_cartridge_only.png"))

# scatter plots of cartridge daily price differences
pair_plot = sns.pairplot(
    master[["9mm", "223", "762x39"]].rename(columns=lambda x: x + " day price change($)").pct_change(1),
    corner=True
    )
# pair_plot.map_lower(corr)
fig = pair_plot.fig
fig.savefig(path_adapt("daily_returns_cartridge_only.png"))

# scatter plots of cci after being logged
master["cci_logged"] = master["cci"] ** 2
# master["cci_logged"] = np.exp2(master["cci"]) 
fig, axs = plt.subplots(3,1)
for j, y in enumerate(["9mm", "223", "762x39"]):
    master.plot.scatter("cci_logged", y, ax=axs[j])
fig.set_figheight(10)
fig.set_figwidth(5)
fig.tight_layout()
fig.savefig(path_adapt("scatters_cci.png"))

# raw scatter plots - after covid
master = master.iloc[-730:]
fig, axs = plt.subplots(3,5)
for j, y in enumerate(["9mm", "223", "762x39"]):
    for i, x in enumerate(["cci", "oil", "gas", "copper", "cotton"]):
        master.plot.scatter(x, y, ax=axs[j, i])
fig.set_figheight(10)
fig.set_figwidth(15)
fig.tight_layout()
fig.savefig(path_adapt("scatters_after_covid.png"))

# scatter plots of daily returns (excluding cci) - after covid
fig, axs = plt.subplots(3,4)
for j, y in enumerate(["9mm", "223", "762x39"]):
    for i, x in enumerate(["oil", "gas", "copper", "cotton"]):
        master.pct_change(1).plot.scatter(x, y, ax=axs[j, i])
fig.set_figheight(10)
fig.set_figwidth(15)
fig.tight_layout()
fig.savefig(path_adapt("daily_returns_after_covid.png"))
