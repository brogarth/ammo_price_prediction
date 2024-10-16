import pandas as pd
import numpy as np
np.float_ = np.float64
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from prophet import Prophet as pro
from prophet.plot import plot_forecast_component, plot_weekly, plot_yearly
from prophet.plot import plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import itertools
import copy
import os

colors = list(mcolors.TABLEAU_COLORS.keys())

# set up output folder
OUTPUT_FOLDER = os.path.join(os.getcwd(), "output")
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
def path_adapt(name: str):
    return os.path.join(OUTPUT_FOLDER, name)

# import ammo data - clean up if needed
def import_proph_data(fi):
    df = pd.read_csv(fi)
    header = list(df.columns)
    df = df.rename(columns={header[0]: "ds", header[1]: "y"})
    df["ds"] = pd.to_datetime(df["ds"], format='%Y-%m-%d %H:%M:%S')
    return df

# import CCI data
def import_CCI_data(fi):
    # a little special: need to replace dates with last tuesday of month
    df = pd.read_csv(fi, header=2, names=("ds", "CCI"))
    df["ds"] = pd.to_datetime(df["ds"], format="%m/%d/%Y %H:%M")
    df["ds"] = df["ds"] + pd.offsets.LastWeekOfMonth(weekday=1)

    # add another month to give some extrapolation room
    buffer = pd.DataFrame([{"ds": df["ds"].iloc[-1] + pd.offsets.LastWeekOfMonth(weekday=1), "CCI": df["CCI"].iloc[-1]}])
    df = pd.concat([df, buffer], ignore_index=True)
    df = df.set_index("ds").resample('D').mean().ffill()
    df.reset_index(inplace=True)
    return df

# import futures data
def import_stock_data(fi, price: str):
    df = pd.read_csv(fi, usecols=("Date", price))
    df = df.rename(columns={"Date":"ds"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d")
    df = df.set_index("ds").resample('D').mean().ffill()
    df.reset_index(inplace=True)
    return df

# just compare plots
def compare(*args, normalize=False):
    master = pd.read_csv(args[0])
    for fi in args[1:]:
        df = pd.read_csv(fi)
        master = pd.merge(master, df)
    cols = master.columns
    master[cols[0]] = pd.to_datetime(df[cols[0]], format='%Y-%m-%d %H:%M:%S')
    master = master.set_index(cols[0])
    if normalize:
        master = master / master.iloc[0]
        # master = master.rolling(10).mean()
        # master = master.ewm(span=10).mean()
        master = master.ewm(alpha=0.1).mean()
    
    # pandas is trash at plotting time series dates correctly
    _, ax = plt.subplots()
    for c in cols[1:]:
        ax.plot(master.index, master[c], label=c)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    # ax = master.plot()
    # x_labels = [date.strftime("%Y-%m-%d") for date in master.index] 
    # ax.set_xticklabels(x_labels)
    # ax.set_xlabel("Date")

    return master, ax

# put together seasonal/trend plots
def multi_plot_components(*args):
    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    axs = (ax1, ax2, ax3)
    for (i, (m,f,l)) in enumerate(args):
        plot_forecast_component(m=m, fcst=f, name="trend", ax=ax1)
        plot_weekly(m=m, ax=ax2)
        plot_yearly(m=m, ax=ax3)
        for ax in axs:
            ax.lines[-1].set_color(colors[i])
            ax.lines[-1].set_label(l)
    ax1.legend()
    return fig
 
# plot a bunch of forecasts at once
def multi_plot_forecasts(*args):
    fig = plt.figure(figsize=(10, 12))
    for (i, (m, f, l)) in enumerate(args):
        ax = fig.add_subplot(len(args), 1, i+1)
        m.plot(f, ax=ax, ylabel=l)
    return fig

# do multiple cross validations and plot them
def multi_plot_cv_metric(metric: str, params: dict, *args):
    fig, ax = plt.subplots()
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric)
    dfs = {}
    for (i, (m, f, l)) in enumerate(args):
        # df_cv = cross_validation(m, **params)
        df_cv = cross_validation(m, **params,  parallel="threads")
        df_mt = performance_metrics(df_cv, rolling_window=1/len(df_cv))
        dfs[l] = df_mt
        df_mt[metric].plot(ax=ax)
        ax.lines[-1].set_label(l)
    ax.legend()
    ax.grid()
    ax.set_ylim(bottom=0.0, top=0.9)
    return fig, dfs

# bar chart of rolling errors
def multi_plot_rolling(metric: str, exp_names: list, horizon: tuple, *args):
    x = np.arange(len(exp_names))
    width = 0.25
    mult = 0
    fig, ax = plt.subplots()
    for l in args[0].keys():
        offset = width * mult
        bars = ax.bar(x + offset, 
                      [x[l][metric][horizon[0]:horizon[1]].mean() for x in args], 
                      width, 
                      label=l
        )
        ax.bar_label(bars, padding = 3)
        mult += 1
    ax.set_xlabel("Experiment")
    ax.set_xticks(x + width, exp_names)
    ax.set_ylabel(metric)
    ax.legend()
    return fig

# grid searching
def grid_search(
        variables: dict, 
        params: dict, 
        df: pd.DataFrame, 
        exp_name: str,
        fixed_vars = None, # pd.DataFrame if using
        holidays = None, # pd.DataFrame if custom, str if just built-in days ('US'),
        add_regressor_seasonality = None, # or 'additive', 'multiplicative'
):
    exp_params = [dict(zip(variables.keys(), v)) for v in itertools.product(*variables.values())]
    mapes = []

    if fixed_vars != None:
        exp_params = [{**fixed_vars, **k} for k in exp_params]
    
    for p in exp_params:
        if type(holidays) == str:
            m = pro(**p)
            m.add_country_holidays(country_name=holidays)
        elif type(holidays) == pd.DataFrame:
            m = pro(**p, holidays=holidays)
        else:
            m = pro(**p)
        if len(df.columns) > 2:
            for c in df.columns[2:]:
                if add_regressor_seasonality != None:
                    m.add_regressor(c, mode=add_regressor_seasonality)
                else:
                    m.add_regressor(c)
        m.fit(df)
        cv = cross_validation(m, **params, parallel="threads")
        pm = performance_metrics(cv, rolling_window=1)
        mapes.append(pm['mape'].values[0])
        if mapes[-1] == np.min(mapes):
            min_model = copy.deepcopy(m) # I don't trust if this is a reference or not
    
    results = pd.DataFrame(exp_params)
    results['mape'] = mapes
    results.to_markdown(path_adapt(f"grid_search_{exp_name}.md"))
    best_params = exp_params[np.argmin(mapes)]

    # return best prophet model
    return min_model, best_params
