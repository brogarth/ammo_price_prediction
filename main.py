from utils import *
from prophet import Prophet as pro
import pickle
from prophet.utilities import regressor_coefficients

# read in data for Prophet
nine = import_proph_data("./data/daily_9mm.csv")
r223 = import_proph_data("./data/daily_223.csv")
x762 = import_proph_data("./data/daily_x39.csv")

# define periods to care about for cross validation
params = {
    "initial": '730 days', 
    "period": '90 days', 
    "horizon": '180 days'
}

# default models - no tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see if results already exist
models_file = path_adapt("models_default_no_tune.pkl")
if not os.path.exists(models_file):
 
    m1 = pro().fit(nine)
    m2 = pro().fit(r223)
    m3 = pro().fit(x762)

    # make some forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_default_no_tune.png"))

    # look at some initial forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_default_no_tune.png"))

    # calculate errors of said forecasts
    fig, dfs_nt = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_default_no_tune.png"))

    # save for later
    models = {
        "nine": (m1, None), 
        "r223": (m2, None), 
        "x762": (m3, None), 
        "errors": dfs_nt
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, _ = models["nine"]
        m2, _ = models["r223"]
        m3, _ = models["x762"]
        dfs_nt = models["errors"]

# default models - WITH tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see if results already exist
models_file = path_adapt("models_default.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_default_bests = grid_search(grid, params, nine, "default_nine")
    m2, m2_default_bests = grid_search(grid, params, r223, "default_r223")
    m3, m3_default_bests = grid_search(grid, params, x762, "default_x762")

    # make some forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_default.png"))

    # look at some initial forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_default.png"))

    # calculate errors of said forecasts
    fig, dfs = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_default.png"))

    # save for later
    models = {
        "nine": (m1, m1_default_bests), 
        "r223": (m2, m2_default_bests), 
        "x762": (m3, m3_default_bests), 
        "errors": dfs
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_default_bests = models["nine"]
        m2, m2_default_bests = models["r223"]
        m3, m3_default_bests = models["x762"]
        dfs = models["errors"]

# add in holidays ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see if results already exist
models_file = path_adapt("models_holidays.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_holidays_bests = grid_search(grid, params, nine, "holidays_nine", m1_default_bests, "US")
    m2, m2_holidays_bests = grid_search(grid, params, r223, "holidays_r223", m2_default_bests, "US")
    m3, m3_holidays_bests = grid_search(grid, params, x762, "holidays_x762", m3_default_bests, "US")

    # remake forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_holidays.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_holidays.png"))

    # calculate errors of said forecasts
    fig, dfs_holidays = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_holidays.png"))

    # save for later
    models = {
        "nine": (m1, m1_holidays_bests), 
        "r223": (m2, m2_holidays_bests), 
        "x762": (m3, m3_holidays_bests), 
        "errors": dfs_holidays
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_holidays_bests = models["nine"]
        m2, m2_holidays_bests = models["r223"]
        m3, m3_holidays_bests = models["x762"]
        dfs_holidays = models["errors"]

# add in world events ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# define special events to use as 1 time/infrequent holidays (covid, wars, elections)
special_days = pd.DataFrame([
        {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
        {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
        {'holiday': '2020_elctn', 'ds': '2020-11-03', 'lower_window': 0, 'ds_upper': '2020-11-03'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
        {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
        {'holiday': 'import_ban', 'ds': '2021-08-20', 'lower_window': 0, 'ds_upper': '2021-09-07'},
        {'holiday': 'Rs/Ukr war', 'ds': '2022-02-24', 'lower_window': 0, 'ds_upper': '2022-02-24'},
        {'holiday': 'Israel Inv', 'ds': '2023-10-07', 'lower_window': 0, 'ds_upper': '2023-10-27'},
        {'holiday': '2024_elctn', 'ds': '2024-11-05', 'lower_window': 0, 'ds_upper': '2024-11-05'}
    ])
for t_col in ['ds', 'ds_upper']:
    special_days[t_col] = pd.to_datetime(special_days[t_col])
special_days['upper_window'] = (special_days['ds_upper'] - special_days['ds']).dt.days

# see if results already exist
models_file = path_adapt("models_events.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_events_bests = grid_search(grid, params, nine, "events_nine", m1_default_bests, special_days)
    m2, m2_events_bests = grid_search(grid, params, r223, "events_r223", m2_default_bests, special_days)
    m3, m3_events_bests = grid_search(grid, params, x762, "events_x762", m3_default_bests, special_days)

    # remake forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_events.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_events.png"))

    # calculate errors of said forecasts
    fig, dfs_events = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_events.png"))

    # save for later
    models = {
        "nine": (m1, m1_events_bests), 
        "r223": (m2, m2_events_bests), 
        "x762": (m3, m3_events_bests), 
        "errors": dfs_events
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

    m1_events = copy.deepcopy(m1)
    m2_events = copy.deepcopy(m2)
    m3_events = copy.deepcopy(m2)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1_events, m1_events_bests = models["nine"]
        m2_events, m2_events_bests = models["r223"]
        m3_events, m3_events_bests = models["x762"]
        dfs_events = models["errors"]

# model with each other's prices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# establish dataframes
nine_combo = nine.copy()
nine_combo = pd.merge(nine_combo, r223.rename(columns={"y":"r223"}), how="left")
nine_combo = pd.merge(nine_combo, x762.rename(columns={"y":"x762"}), how="left")

r223_combo = r223.copy()
r223_combo = pd.merge(r223_combo, nine.rename(columns={"y":"nine"}), how="left")
r223_combo = pd.merge(r223_combo, x762.rename(columns={"y":"x762"}), how="left")

x762_combo = x762.copy()
x762_combo = pd.merge(x762_combo, nine.rename(columns={"y":"nine"}), how="left")
x762_combo = pd.merge(x762_combo, r223.rename(columns={"y":"r223"}), how="left")

# see if results already exist
models_file = path_adapt("models_combo.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    key = "holidays_prior_scale"
    m1, m1_combo_bests = grid_search(grid, params, nine_combo, "combo_nine", {key: m1_events_bests[key]}, special_days)
    m2, m2_combo_bests = grid_search(grid, params, r223_combo, "combo_r223", {key: m2_events_bests[key]}, special_days)
    m3, m3_combo_bests = grid_search(grid, params, x762_combo, "combo_x762", {key: m3_events_bests[key]}, special_days)

    # remake forecasts - re-use predictions from prior models for predictions of other regressors
    sixmo_sched = m1.make_future_dataframe(periods=180)
    sixmo_sched = pd.merge(sixmo_sched, nine_combo.rename(columns={"y":"nine"}), how="left")
    sixmo_sched.loc[sixmo_sched.tail(180).index, "nine"] = m1_fcst["yhat"].iloc[-180:]
    sixmo_sched.loc[sixmo_sched.tail(180).index, "r223"] = m2_fcst["yhat"].iloc[-180:]
    sixmo_sched.loc[sixmo_sched.tail(180).index, "x762"] = m2_fcst["yhat"].iloc[-180:]

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_combo.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_combo.png"))

    # calculate errors of said forecasts
    fig, dfs_combo = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_combo.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("combo_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("combo_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("combo_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_combo_bests), 
        "r223": (m2, m2_combo_bests), 
        "x762": (m3, m3_combo_bests), 
        "errors": dfs_combo
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_combo_bests = models["nine"]
        m2, m2_combo_bests = models["r223"]
        m3, m3_combo_bests = models["x762"]
        dfs_combo = models["errors"]

# # try a model with a bunch of factors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # establish dataframes
# datas = [import_CCI_data("data/CCI_US.csv")]
# nine_all = pd.merge(nine, datas[-1], how="left")
# r223_all = pd.merge(r223, datas[-1], how="left")
# x762_all = pd.merge(x762, datas[-1], how="left")
# names = ["CCI", "crude", "cotton", "copper", "nat_gas"]

# # see if results already exist
# models_file = path_adapt("models_factors.pkl")
# if not os.path.exists(models_file):

#     for (i, fi) in enumerate(("data/CL=F.csv", "data/CT=F.csv", "data/HG=F.csv", "data/NG=F.csv")):
#     # for (i, fi) in enumerate(["data/HG=F.csv",]):
#         futures = import_stock_data(fi, "Close")
#         futures = futures.rename(columns={"Close": names[i+1]})
#         datas.append(futures)
#         nine_all = pd.merge(nine_all, futures, how="left")
#         r223_all = pd.merge(r223_all, futures, how="left")
#         x762_all = pd.merge(x762_all, futures, how="left")

#     # Do another grid search since the model is changing
#     grid = {  
#         'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
#         'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     }

#     m1, m1_factors_bests = grid_search(grid, params, nine_all, "factors_nine", {key: m1_events_bests[key]}, special_days)
#     m2, m2_factors_bests = grid_search(grid, params, r223_all, "factors_r223", {key: m2_events_bests[key]}, special_days)
#     m3, m3_factors_bests = grid_search(grid, params, x762_all, "factors_x762", {key: m3_events_bests[key]}, special_days)

#     # remake forecasts
#     sixmo_sched = m1.make_future_dataframe(periods=180)

#     # need to extrapolate additional regressors for future predictions - shouldn't matter for cv
#     for (i, n) in enumerate(names):
#         m = pro()
#         m.fit(datas[i].iloc[-len(nine_all):].rename(columns={n:"y"}))
#         m_fcst = m.predict(sixmo_sched)
#         sixmo_sched = pd.merge(sixmo_sched, datas[i], how="left")
#         sixmo_sched.loc[sixmo_sched.tail(180).index, n] = m_fcst["yhat"].iloc[-180:]

#     m1_fcst = m1.predict(sixmo_sched)
#     m2_fcst = m2.predict(sixmo_sched)
#     m3_fcst = m3.predict(sixmo_sched)

#     multi_args = (
#         (m1, m1_fcst, "9x19"), 
#         (m2, m2_fcst, ".223 R"), 
#         (m3, m3_fcst, "7.62x39")
#     )

#     # look at combined components
#     fig = multi_plot_components(*multi_args)
#     fig.savefig(path_adapt("multi_components_factors.png"))

#     # view the forecasts
#     fig = multi_plot_forecasts(*multi_args)
#     fig.savefig(path_adapt("multi_forecast_factors.png"))

#     # calculate errors of said forecasts
#     fig, dfs_factors = multi_plot_cv_metric("mape", params, *multi_args)
#     fig.savefig(path_adapt("multi_mape_factors.png"))

#     # save coefficents of extra regressors
#     regressor_coefficients(m1).to_markdown(path_adapt("factors_nine_coefficents.md"))
#     regressor_coefficients(m2).to_markdown(path_adapt("factors_r223_coefficents.md"))
#     regressor_coefficients(m3).to_markdown(path_adapt("factors_x762_coefficents.md"))

#     # save for later
#     models = {
#         "nine": (m1, m1_factors_bests), 
#         "r223": (m2, m2_factors_bests), 
#         "x762": (m3, m3_factors_bests), 
#         "errors": dfs_factors
#         }
#     with open(models_file, "wb") as f:
#         pickle.dump(models, f)

# else:
#     # unpack from file for potential re-use
#     with open(models_file, "rb") as f:
#         models = pickle.load(f)
#         m1, m1_factors_bests = models["nine"]
#         m2, m2_factors_bests = models["r223"]
#         m3, m3_factors_bests = models["x762"]
#         dfs_factors = models["errors"]

# build a model using cci data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# establish dataframes
cci = import_CCI_data("data/CCI_US.csv")
nine_cci = pd.merge(nine, cci, how="left")
r223_cci = pd.merge(r223, cci, how="left")
x762_cci = pd.merge(x762, cci, how="left")

# see if results already exist
models_file = path_adapt("models_cci.pkl")
if not os.path.exists(models_file):

    # Do another grid search since the model is changing
    grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_cci_bests = grid_search(grid, params, nine_cci, "cci_nine")
    m2, m2_cci_bests = grid_search(grid, params, r223_cci, "cci_r223")
    m3, m3_cci_bests = grid_search(grid, params, x762_cci, "cci_x762")

    # remake forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    # need to extrapolate additional regressors for future predictions - shouldn't matter for cv
    m = pro()
    m.fit(cci.iloc[-len(nine_cci):].rename(columns={"CCI":"y"}))
    m_fcst = m.predict(sixmo_sched)
    sixmo_sched = pd.merge(sixmo_sched, cci, how="left")
    sixmo_sched.loc[sixmo_sched.tail(180).index, "CCI"] = m_fcst["yhat"].iloc[-180:]

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_cci.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_cci.png"))

    # calculate errors of said forecasts
    fig, dfs_cci = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_cci.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("cci_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("cci_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("cci_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_cci_bests), 
        "r223": (m2, m2_cci_bests), 
        "x762": (m3, m3_cci_bests), 
        "errors": dfs_cci
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_cci_bests = models["nine"]
        m2, m2_cci_bests = models["r223"]
        m3, m3_cci_bests = models["x762"]
        dfs_cci = models["errors"]

# combine events + cci data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see if results already exist
models_file = path_adapt("models_events_and_cci.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_events_and_cci_bests = grid_search(grid, params, nine_cci, "events_and_cci_nine", m1_cci_bests, special_days)
    m2, m2_events_and_cci_bests = grid_search(grid, params, r223_cci, "events_and_cci_r223", m2_cci_bests, special_days)
    m3, m3_events_and_cci_bests = grid_search(grid, params, x762_cci, "events_and_cci_x762", m3_cci_bests, special_days)

    # remake forecasts
    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_events_and_cci.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_events_and_cci.png"))

    # calculate errors of said forecasts
    fig, dfs_events_and_cci = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_events_and_cci.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("events_and_cci_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("events_and_cci_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("events_and_cci_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_events_and_cci_bests), 
        "r223": (m2, m2_events_and_cci_bests), 
        "x762": (m3, m3_events_and_cci_bests), 
        "errors": dfs_events_and_cci
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_events_and_cci_bests = models["nine"]
        m2, m2_events_and_cci_bests = models["r223"]
        m3, m3_events_and_cci_bests = models["x762"]
        dfs_events_and_cci = models["errors"]

# build a model using oil data - treat < $65/barrel as a low signal ~~~~~~~~~~~~

# establish dataframes
oil = import_stock_data("data/CL=F.csv", "Close")
oil = oil.rename(columns={"Close": "crude"})
oil["crude"] = oil["crude"] < 70
oil["crude"] = oil["crude"].astype(int)

nine_oil = pd.merge(nine, oil, how="left")
r223_oil = pd.merge(r223, oil, how="left")
x762_oil = pd.merge(x762, oil, how="left")

# see if results already exist
models_file = path_adapt("models_oil.pkl")
if not os.path.exists(models_file):

    # Do another grid search since the model is changing
    grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_oil_bests = grid_search(grid, params, nine_oil, "oil_nine", add_regressor_seasonality="multiplicative")
    m2, m2_oil_bests = grid_search(grid, params, r223_oil, "oil_r223", add_regressor_seasonality="multiplicative")
    m3, m3_oil_bests = grid_search(grid, params, x762_oil, "oil_x762", add_regressor_seasonality="multiplicative")

    # remake forecasts
    sixmo_sched = m1.make_future_dataframe(periods=180)

    # need to extrapolate additional regressors for future predictions - shouldn't matter for cv
    m = pro()
    m.fit(oil.iloc[-len(nine_oil):].rename(columns={"crude":"y"}))
    m_fcst = m.predict(sixmo_sched)
    sixmo_sched = pd.merge(sixmo_sched, oil, how="left")
    sixmo_sched.loc[sixmo_sched.tail(180).index, "crude"] = m_fcst["yhat"].iloc[-180:]

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_oil.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_oil.png"))

    # calculate errors of said forecasts
    fig, dfs_oil = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_oil.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("oil_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("oil_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("oil_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_oil_bests), 
        "r223": (m2, m2_oil_bests), 
        "x762": (m3, m3_oil_bests), 
        "errors": dfs_oil
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_oil_bests = models["nine"]
        m2, m2_oil_bests = models["r223"]
        m3, m3_oil_bests = models["x762"]
        dfs_oil = models["errors"]

# combine events + oil data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# see if results already exist
models_file = path_adapt("models_events_and_oil.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    m1, m1_events_and_oil_bests = grid_search(grid, params, nine_oil, "events_and_oil_nine", m1_oil_bests, special_days, "multiplicative")
    m2, m2_events_and_oil_bests = grid_search(grid, params, r223_oil, "events_and_oil_r223", m2_oil_bests, special_days, "multiplicative")
    m3, m3_events_and_oil_bests = grid_search(grid, params, x762_oil, "events_and_oil_x762", m3_oil_bests, special_days, "multiplicative")

    # remake forecasts
    # sixmo_sched = m1.make_future_dataframe(periods=180)

    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_events_and_oil.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_events_and_oil.png"))

    # calculate errors of said forecasts
    fig, dfs_events_and_oil = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_events_and_oil.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("events_and_oil_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("events_and_oil_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("events_and_oil_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_events_and_oil_bests), 
        "r223": (m2, m2_events_and_oil_bests), 
        "x762": (m3, m3_events_and_oil_bests), 
        "errors": dfs_events_and_oil
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

    m1_oil = copy.deepcopy(m1)
    m2_oil = copy.deepcopy(m2)
    m3.oil = copy.deepcopy(m3)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1_oil, m1_events_and_oil_bests = models["nine"]
        m2_oil, m2_events_and_oil_bests = models["r223"]
        m3_oil, m3_events_and_oil_bests = models["x762"]
        dfs_events_and_oil = models["errors"]

# one last model: oil + events + each others prices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# establish dataframes
nine_mixed = pd.merge(nine_combo, oil, how="left")
r223_mixed = pd.merge(r223_combo, oil, how="left")
x762_mixed = pd.merge(x762_combo, oil, how="left")

# see if results already exist
models_file = path_adapt("models_mixed.pkl")
if not os.path.exists(models_file):

    # do some grid searching for good starting parameters
    grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.05, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    key = "holidays_prior_scale"
    m1, m1_mixed_bests = grid_search(grid, params, nine_mixed, "mixed_nine", {key: m1_events_bests[key]}, special_days)
    m2, m2_mixed_bests = grid_search(grid, params, r223_mixed, "mixed_r223", {key: m2_events_bests[key]}, special_days)
    m3, m3_mixed_bests = grid_search(grid, params, x762_mixed, "mixed_x762", {key: m3_events_bests[key]}, special_days)

    # remake forecasts - re-use predictions from prior models for predictions of other regressors
    sixmo_sched = m1.make_future_dataframe(periods=180)
    sixmo_sched = pd.merge(sixmo_sched, nine_mixed.rename(columns={"y":"nine"}), how="left")

    sixmo_sched.loc[sixmo_sched.tail(180).index, "nine"] = m1_events.predict(sixmo_sched)["yhat"].iloc[-180:]
    sixmo_sched.loc[sixmo_sched.tail(180).index, "r223"] = m2_events.predict(sixmo_sched)["yhat"].iloc[-180:]
    sixmo_sched.loc[sixmo_sched.tail(180).index, "x762"] = m3_events.predict(sixmo_sched)["yhat"].iloc[-180:]
    
    m = pro()
    m.fit(nine_mixed[["ds", "crude"]].rename(columns={"crude":"y"}))
    m_fcst = m.predict(sixmo_sched)
    sixmo_sched.loc[sixmo_sched.tail(180).index, "crude"] = m_fcst["yhat"].iloc[-180:]


    m1_fcst = m1.predict(sixmo_sched)
    m2_fcst = m2.predict(sixmo_sched)
    m3_fcst = m3.predict(sixmo_sched)

    multi_args = (
        (m1, m1_fcst, "9x19"), 
        (m2, m2_fcst, ".223 R"), 
        (m3, m3_fcst, "7.62x39")
    )

    # look at combined components
    fig = multi_plot_components(*multi_args)
    fig.savefig(path_adapt("multi_components_mixed.png"))

    # view the forecasts
    fig = multi_plot_forecasts(*multi_args)
    fig.savefig(path_adapt("multi_forecast_mixed.png"))

    # calculate errors of said forecasts
    fig, dfs_mixed = multi_plot_cv_metric("mape", params, *multi_args)
    fig.savefig(path_adapt("multi_mape_mixed.png"))

    # save coefficents of extra regressors
    regressor_coefficients(m1).to_markdown(path_adapt("mixed_nine_coefficents.md"))
    regressor_coefficients(m2).to_markdown(path_adapt("mixed_r223_coefficents.md"))
    regressor_coefficients(m3).to_markdown(path_adapt("mixed_x762_coefficents.md"))

    # save for later
    models = {
        "nine": (m1, m1_mixed_bests), 
        "r223": (m2, m2_mixed_bests), 
        "x762": (m3, m3_mixed_bests), 
        "errors": dfs_mixed
        }
    with open(models_file, "wb") as f:
        pickle.dump(models, f)

else:
    # unpack from file for potential re-use
    with open(models_file, "rb") as f:
        models = pickle.load(f)
        m1, m1_mixed_bests = models["nine"]
        m2, m2_mixed_bests = models["r223"]
        m3, m3_mixed_bests = models["x762"]
        dfs_mixed = models["errors"]

# compare model performances ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# save for later
exps = ["no_tune", "default", "holidays", "events", "events+eachother", "cci", "cci+events", "oil", "oil+events", "mixed"]
dfss = [dfs_nt, dfs, dfs_holidays, dfs_events, dfs_combo, dfs_cci, dfs_events_and_cci, dfs_oil, dfs_events_and_oil, dfs_mixed]

model_results = {e:d for e,d in zip(exps, dfss)}
with open(path_adapt("model_results.pkl"), "wb") as f:
    pickle.dump(model_results, f)

# just using pricing data
fig = multi_plot_rolling("mape", exps[0:5], (0, 60), *dfss[0:5])
fig.savefig(path_adapt("mean_mapes_ammo_only_0_to_60_days.png"))
fig = multi_plot_rolling("mape", exps[0:5], (60, 180), *dfss[0:5])
fig.savefig(path_adapt("mean_mapes__ammo_only_60_to_180_days.png"))

# using factors data
exps = [exps[1]] + exps[5:]
dfss = [dfss[1]] + dfss[5:]

fig = multi_plot_rolling("mape", exps, (0, 60), *dfss)
fig.savefig(path_adapt("mean_mapes_w_factors_0_to_60_days.png"))
fig = multi_plot_rolling("mape", exps, (60, 180), *dfss)
fig.savefig(path_adapt("mean_mapes_w_factors_60_to_180_days.png"))