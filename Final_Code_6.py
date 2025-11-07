# SAIL 2025 – Crowd Flow Dashboard 

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import folium
from streamlit_folium import st_folium
import joblib
import json
from datetime import datetime, timedelta

#page configuration
st.set_page_config(page_title="SAIL 2025 – Sensor Map", layout="wide")
st.title("SAIL Amsterdam Sensor Map")

#keep margins small so map+charts fit better
st.markdown("""
<style>
.block-container{padding:0.6rem;}
h1,h2,h3{margin:0.2rem 0;}
.stSlider{padding:0.25rem 0;}
</style>
""", unsafe_allow_html=True)


#file paths
FLOW_CSV = r"sensor_data_modified.csv"
LOC_CSV  = r"sensor_location_Ines.csv"
QMAX_CSV = r"Q_Max.csv"
# SAIL 2025 – Crowd Flow Dashboard 


#colours
LOS_COLORS = {
    "A": "#2ECC40",   # green
    "B": "#FFD60A",   # yellow
    "C": "#FFA500",   # light orange
    "D": "#FF8C00",   # orange
    "E": "#FF4D4F",   # red
    "F": "#8E44AD"    # purple
}
LOS_COLOR_LIST = [LOS_COLORS[k] for k in "ABCDEF"]

# FORECASTING MODEL FUNCTIONS
##############################################################################

def load_forecast_model(model_path="sensor_forecast_model"):
    """Load the trained model and all components"""
    try:
        model = joblib.load(f"{model_path}/xgboost_model.joblib")
        scaler = joblib.load(f"{model_path}/scaler.joblib")
        
        with open(f"{model_path}/feature_cols.json", 'r') as f:
            feature_cols = json.load(f)
            
        with open(f"{model_path}/sensor_cols.json", 'r') as f:
            sensor_cols = json.load(f)
        
        st.success(f"✅ Model loaded successfully for {len(sensor_cols)} sensors")
        return model, scaler, feature_cols, sensor_cols
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

def aggregate_to_15min(df_3min):
    """Aggregate 3-minute data to 15-minute intervals"""
    # Get sensor columns (exclude time features)
    sensor_cols = [col for col in df_3min.columns if col not in ['hour', 'minute', 'day', 'month', 'weekday', 'is_weekend']]
    
    # Aggregate sensor data - use 'min' instead of deprecated 'T'
    df_15min = df_3min[sensor_cols].resample('15min').sum()
    
    # Recreate time features
    df_15min['hour'] = df_15min.index.hour
    df_15min['minute'] = df_15min.index.minute
    df_15min['day'] = df_15min.index.day
    df_15min['month'] = df_15min.index.month
    df_15min['weekday'] = df_15min.index.weekday
    df_15min['is_weekend'] = df_15min.index.weekday.isin([5, 6]).astype(int)
    
    return df_15min

def create_lag_features_fixed(df, sensor_cols, lag_periods=[1, 2, 3, 4, 96, 97, 98]):
    """Create lagged features using ACTUAL historical data - completely fixed version"""
    df_lagged = df.copy()
    
    # Create lag features for each sensor column
    for col in sensor_cols:
        if col in df_lagged.columns:
            for lag in lag_periods:
                lag_col_name = f'{col}_lag_{lag}'
                df_lagged[lag_col_name] = df_lagged[col].shift(lag)
    
    # Create rolling statistics for each sensor column
    for col in sensor_cols:
        if col in df_lagged.columns:
            df_lagged[f'{col}_rolling_mean_4'] = df_lagged[col].rolling(window=4, min_periods=1).mean()
            df_lagged[f'{col}_rolling_std_4'] = df_lagged[col].rolling(window=4, min_periods=1).std()
            df_lagged[f'{col}_rolling_mean_96'] = df_lagged[col].rolling(window=96, min_periods=1).mean()
    
    # Fill NaN values
    df_lagged = df_lagged.fillna(method='bfill').fillna(0)
    
    return df_lagged

def create_cyclical_features(df):
    """Convert time features to cyclical format"""
    df_cyclic = df.copy()
    
    # Ensure all time features are numeric, not methods
    for col in ['hour', 'minute', 'day', 'month', 'weekday']:
        if col in df_cyclic.columns:
            # Convert to numeric, coercing any non-numeric to NaN then filling
            df_cyclic[col] = pd.to_numeric(df_cyclic[col], errors='coerce').fillna(0)
    
    # Cyclical encoding for hour and minute
    df_cyclic['hour_sin'] = np.sin(2 * np.pi * df_cyclic['hour'] / 24)
    df_cyclic['hour_cos'] = np.cos(2 * np.pi * df_cyclic['hour'] / 24)
    df_cyclic['minute_sin'] = np.sin(2 * np.pi * df_cyclic['minute'] / 60)
    df_cyclic['minute_cos'] = np.cos(2 * np.pi * df_cyclic['minute'] / 60)
    
    # Day of week cyclical
    df_cyclic['weekday_sin'] = np.sin(2 * np.pi * df_cyclic['weekday'] / 7)
    df_cyclic['weekday_cos'] = np.cos(2 * np.pi * df_cyclic['weekday'] / 7)
    
    # Month cyclical
    df_cyclic['month_sin'] = np.sin(2 * np.pi * df_cyclic['month'] / 12)
    df_cyclic['month_cos'] = np.cos(2 * np.pi * df_cyclic['month'] / 12)
    
    return df_cyclic

def ensure_feature_columns(df, feature_cols):
    """
    Ensure the DataFrame has exactly the features the model expects, in the right order
    """
    # Create a new DataFrame with the exact feature columns in the right order
    result_df = pd.DataFrame(index=df.index)
    
    for col in feature_cols:
        if col in df.columns:
            result_df[col] = df[col]
        else:
            # If feature is missing, fill with 0
            result_df[col] = 0
    
    return result_df

def update_features_with_forecast(df, forecast_values, sensor_cols, forecast_time, feature_cols):
    """
    Update the DataFrame with new forecast values and recompute lag features
    """
    # Create new row with forecast values
    new_row = pd.DataFrame([forecast_values], columns=sensor_cols, index=[forecast_time])
    
    # Add time features for the new time - ensure they are numeric values, not methods
    new_row['hour'] = float(forecast_time.hour)
    new_row['minute'] = float(forecast_time.minute)
    new_row['day'] = float(forecast_time.day)
    new_row['month'] = float(forecast_time.month)
    new_row['weekday'] = float(forecast_time.weekday())
    new_row['is_weekend'] = 1.0 if forecast_time.weekday() in [5, 6] else 0.0
    
    # Append to DataFrame
    df_updated = pd.concat([df, new_row])
    
    # Recompute lag features for the entire updated DataFrame
    df_with_lags = create_lag_features_fixed(df_updated, sensor_cols)
    df_with_features = create_cyclical_features(df_with_lags)
    
    # Ensure we have exactly the features the model expects, in the right order
    df_with_features = ensure_feature_columns(df_with_features, feature_cols)
    
    return df_with_features

def prepare_features_for_time(df, feature_cols, target_time):
    """
    Extract features for a specific time from the prepared DataFrame
    """
    if target_time in df.index:
        features = df.loc[[target_time]]
    else:
        # If time not found, use the latest available
        features = df.iloc[[-1]]
    
    # Ensure we have exactly the right columns in the right order
    features = ensure_feature_columns(features, feature_cols)
    
    return features

def forecast_next_2_hours(model, scaler, feature_cols, sensor_cols, recent_data, forecast_datetime):
    """
    Forecast the next 2 hours (8 steps) using RECURSIVE forecasting
    """
    # Remove timezone from data if present
    recent_data = recent_data.copy()
    if recent_data.index.tz is not None:
        recent_data.index = recent_data.index.tz_localize(None)
    
    # Ensure forecast_datetime is timezone-naive
    if hasattr(forecast_datetime, 'tzinfo') and forecast_datetime.tzinfo is not None:
        forecast_datetime = forecast_datetime.replace(tzinfo=None)
    
    # Filter data to ONLY include data BEFORE forecast_datetime
    historical_data = recent_data[recent_data.index < forecast_datetime]
    
    if len(historical_data) == 0:
        raise ValueError(f"No historical data found before {forecast_datetime}")
    
    # Create future timestamps for the forecast period - use 'min' instead of 'T'
    future_dates = pd.date_range(
        start=forecast_datetime,
        periods=8,  # 2 hours = 8 steps of 15min
        freq='15min'
    )
    
    # Prepare initial features using ACTUAL historical data
    df_with_features = create_lag_features_fixed(historical_data, sensor_cols)
    df_with_features = create_cyclical_features(df_with_features)
    
    # Ensure we have exactly the features the model expects
    df_with_features = ensure_feature_columns(df_with_features, feature_cols)
    
    # Ensure all data is numeric
    for col in df_with_features.columns:
        df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce').fillna(0)
    
    all_forecasts = []
    current_df = df_with_features
    
    # Recursive forecasting: predict one step, update features, predict next step
    for i, forecast_time in enumerate(future_dates):
        # Prepare features for current forecast time
        current_features = prepare_features_for_time(current_df, feature_cols, current_df.index[-1])
        
        # Scale features and make prediction
        features_scaled = scaler.transform(current_features)
        forecast = model.predict(features_scaled)
        forecast = np.maximum(forecast, 0)  # Ensure non-negative
        
        # Store forecast
        all_forecasts.append(forecast[0])
        
        # Update DataFrame with this forecast for the next prediction
        if i < len(future_dates) - 1:  # Don't update after the last prediction
            current_df = update_features_with_forecast(
                current_df, 
                forecast[0], 
                sensor_cols, 
                forecast_time,
                feature_cols
            )
    
    # Create results DataFrame
    forecast_results = pd.DataFrame(all_forecasts, columns=sensor_cols, index=future_dates)
    
    return forecast_results

def prepare_forecast_data(df_all, forecast_datetime, model_sensor_cols):
    """
    Prepare data for forecasting from the main application dataframe
    """
    # Create a wide format dataframe similar to Sensor_data.csv
    df_wide = df_all.pivot_table(
        index='timestamp', 
        columns='SensorID', 
        values='count_3min', 
        aggfunc='sum'
    ).fillna(0)
    
    # Ensure all model sensor columns are present, fill missing ones with 0
    for sensor in model_sensor_cols:
        if sensor not in df_wide.columns:
            df_wide[sensor] = 0
    
    # Reorder columns to match model expectation
    df_wide = df_wide.reindex(columns=model_sensor_cols, fill_value=0)
    
    # Add time features
    df_wide['hour'] = df_wide.index.hour
    df_wide['minute'] = df_wide.index.minute
    df_wide['day'] = df_wide.index.day
    df_wide['month'] = df_wide.index.month
    df_wide['weekday'] = df_wide.index.weekday
    df_wide['is_weekend'] = df_wide.index.weekday.isin([5, 6]).astype(int)
    
    # Ensure all time features are numeric
    for col in ['hour', 'minute', 'day', 'month', 'weekday', 'is_weekend']:
        df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce').fillna(0)
    
    # Aggregate to 15-minute intervals
    df_15min = aggregate_to_15min(df_wide)
    
    return df_15min

def get_forecasts_for_sensor(df_all, model, scaler, feature_cols, sensor_cols, sensor_base, forecast_datetime):
    """
    Get forecasts for a specific sensor base (both directions) - CONVERTED TO FLOW PER MINUTE
    """
    try:
        # Check if this sensor base exists in the model's sensor columns
        sensor_directions = [col for col in sensor_cols if col.startswith(sensor_base + '_')]
        
        if not sensor_directions:
            st.warning(f"No forecast model available for sensor {sensor_base}")
            return {}
        
        # Prepare data for forecasting
        forecast_data = prepare_forecast_data(df_all, forecast_datetime, sensor_cols)
        
        # Check if we have enough historical data
        historical_data = forecast_data[forecast_data.index < forecast_datetime]
        if len(historical_data) < 4:  # Need at least some history
            st.warning(f"Insufficient historical data for forecasting sensor {sensor_base}")
            return {}
        
        # Generate forecasts (these are in flow per 15 minutes)
        forecasts = forecast_next_2_hours(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
            recent_data=forecast_data,
            forecast_datetime=forecast_datetime
        )
        
        # Filter forecasts for the specific sensor base directions and CONVERT TO FLOW PER MINUTE
        sensor_forecasts = {}
        for sensor_col in sensor_directions:
            if sensor_col in forecasts.columns:
                direction = sensor_col.split('_')[-1]
                # Convert from flow per 3 minutes to flow per minute by dividing by 3
                sensor_forecasts[direction] = forecasts[sensor_col] / 3.0
        
        return sensor_forecasts
        
    except Exception as e:
        st.error(f"Error generating forecasts for {sensor_base}: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return {}

def get_all_forecasts(df_all, model, scaler, feature_cols, sensor_cols, forecast_datetime):
    """Get forecasts for all sensors"""
    try:
        # Prepare data for forecasting
        forecast_data = prepare_forecast_data(df_all, forecast_datetime, sensor_cols)
        
        # Check if we have enough historical data
        historical_data = forecast_data[forecast_data.index < forecast_datetime]
        if len(historical_data) < 4:
            st.warning("Insufficient historical data for forecasting")
            return {}
        
        # Generate forecasts for all sensors
        forecasts = forecast_next_2_hours(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            sensor_cols=sensor_cols,
            recent_data=forecast_data,
            forecast_datetime=forecast_datetime
        )
        
        # Organize forecasts by sensor base
        all_sensor_forecasts = {}
        sensor_bases = set([col.rsplit('_', 1)[0] for col in sensor_cols])
        
        for sensor_base in sensor_bases:
            sensor_directions = [col for col in sensor_cols if col.startswith(sensor_base + '_')]
            if sensor_directions:
                sensor_forecasts = {}
                for sensor_col in sensor_directions:
                    if sensor_col in forecasts.columns:
                        direction = sensor_col.split('_')[-1]
                        # Convert from flow per 15 minutes to flow per minute
                        sensor_forecasts[direction] = forecasts[sensor_col] / 3.0
                
                if sensor_forecasts:
                    all_sensor_forecasts[sensor_base] = sensor_forecasts
        
        return all_sensor_forecasts
        
    except Exception as e:
        st.error(f"Error generating forecasts: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return {}

# [Keep all your existing functions from the original code unchanged...]
# FUNCTIONS:
##############################################################################
#clean every sensor name: use normal "-", and no extra spaces
def normalize_name(s):
    if s is None: return None
    s = str(s).replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")
    return re.sub(r"\s+"," ", s).strip()

# takes the number at the end of sensor name,
# uses it to determine arrow degrees
def extract_direction_deg(sensor_id):
    m = re.search(r"_([0-9]+)$", str(sensor_id))
    return float(m.group(1)) if m else None


#read, flow CSV. Turns it into table with one row/sensor/time:
# timestamp | SensorID | count_3min | SensorBase | direction_deg
def load_flow_long(path):
    flow = pd.read_csv(path)
    #wide table to long table
    flow_long = flow.melt(id_vars=["timestamp"], var_name="SensorID", value_name="count_3min")
    flow_long["timestamp"] = pd.to_datetime(flow_long["timestamp"], errors="coerce")
    flow_long = flow_long.dropna(subset=["timestamp"])
    flow_long["SensorBase"]   = flow_long["SensorID"].str.replace(r"_[0-9]+$", "", regex=True)
    flow_long["direction_deg"] = flow_long["SensorID"].apply(extract_direction_deg)
    return flow_long

#reads sensor locations CSV. Keeps only needed columns with clean names and coordinates:
# SensorBase | Locatienaam | latitude | longitude 
def load_locations(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    #spelling mistake fix:
    if "Objectnummer" in df.columns:
        key_col = "Objectnummer" 
    elif "Objectummer" in df.columns:
        key_col = "Objectummer"
    else:
        st.error("No 'Objectnummer' or 'Objectummer' column in the locations CSV.")
        st.stop()
    #rename to consistent names   
    df = df.rename(columns={key_col: "SensorBase", "Lat": "latitude", "Long": "longitude"})
    #turn to numeric
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    #keep only needed columns
    out = df[["SensorBase","Locatienaam","latitude","longitude"]].drop_duplicates("SensorBase")
    return out

#reads Qmax LOS A-F CSV and fixes column name
def load_qmax(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    #spelling mistake fix again:
    if "Objectummer" in df.columns:
        key_q = "Objectummer"
    elif "Objectnummer" in df.columns:
        key_q = "Objectnummer"
    else:
        st.error("Qmax CSV missing 'Objectummer' or 'Objectnummer'.")
        st.stop()
    df = df.rename(columns={key_q: "SensorBase"})
    need = ["SensorBase",
            "Q_max_pers_per_min_LOS_A",
            "Q_max_pers_per_min_LOS_B",
            "Q_max_pers_per_min_LOS_C",
            "Q_max_pers_per_min_LOS_D",
            "Q_max_pers_per_min_LOS_E",
            "Q_max_pers_per_min_LOS_F"]
    return df[[c for c in need if c in df.columns]].drop_duplicates("SensorBase")

#combines flow+location+capacity
def build_df_all(flow_long, locs, qmax):
    df = flow_long.merge(locs, on="SensorBase", how="left")
    df = df.merge(qmax, on="SensorBase", how="left")
    #make sure lat/lon are numbers
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    #remove sensors without coordinates
    df = df.dropna(subset=["latitude","longitude"])
    return df


#Chooses accurate LOS-letter, based on measured flow (per sensor at specific moment) 
def los_from_row(r):
    f = r["flow_per_min_used"]
    A = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_A"), errors="coerce")
    B = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_B"), errors="coerce")
    C = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_C"), errors="coerce")
    D = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_D"), errors="coerce")
    E = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_E"), errors="coerce")
    F = pd.to_numeric(r.get("Q_max_pers_per_min_LOS_F"), errors="coerce")
    if pd.isna(f) or pd.isna(A): 
        return None
    if f <= A: return "A"
    if f <= B: return "B"
    if f <= C: return "C"
    if f <= D: return "D"
    if f <= E: return "E"
    return "F"

#gets last 60 minutes of data for a sensor and interpolates missing timestamps, so the trend is continuous.
def get_last_minutes_by_direction(df_all, sensor_base, end_ts, minutes=60):
    if sensor_base is None:
        return pd.DataFrame(columns=["timestamp","direction_deg","flow_per_min_used"])
    #time window
    t_from = pd.Timestamp(end_ts) - pd.Timedelta(minutes=minutes-1)
    #select only needed rows for this sensor and window
    take = df_all[(df_all["SensorBase"]==sensor_base) &
                  (df_all["timestamp"]>=t_from) &
                  (df_all["timestamp"]<=end_ts)].copy()
    #choose which flow to use
    if "flow_per_min_smooth" in take.columns and take["flow_per_min_smooth"].notna().any():
        take["flow_per_min_used"] = pd.to_numeric(take["flow_per_min_smooth"], errors="coerce")
    elif "flow_per_min" in take.columns and take["flow_per_min"].notna().any():
        take["flow_per_min_used"] = pd.to_numeric(take["flow_per_min"], errors="coerce")
    else:
        take["flow_per_min_used"] = pd.to_numeric(take["count_3min"], errors="coerce")/3.0
    #group per (minute, direction)
    g = (take.groupby(["timestamp","direction_deg"], as_index=False)["flow_per_min_used"]
              .sum().sort_values(["timestamp","direction_deg"]))
    #make sure each directioni has a full minute index and interpolate missing minutes
    out = []
    for d in g["direction_deg"].dropna().unique():
        s = g[g["direction_deg"]==d].set_index("timestamp")
        full_idx = pd.date_range(start=(pd.Timestamp(end_ts)-pd.Timedelta(minutes=minutes-1)).floor("min"),
                                 end=pd.Timestamp(end_ts).floor("min"), freq="1min")
        s = s.reindex(full_idx)
        s["direction_deg"] = d
        s["flow_per_min_used"] = s["flow_per_min_used"].interpolate(limit_direction="both")
        s = s.rename_axis("timestamp").reset_index().rename(columns={"index":"timestamp"})
        out.append(s)
    if out: 
        return pd.concat(out, ignore_index=True) 
    else:
        return pd.DataFrame(columns=["timestamp","direction_deg","flow_per_min_used"])

#get LOS lines (A-F) for one sensor
def get_los_lines(df_all, sensor_name):
    cols = ["Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
            "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]
    row = df_all[df_all["SensorBase"]==sensor_name][cols].dropna(how="all").head(1)
    if row.empty: 
        return pd.DataFrame(columns=["LOS","y"])
    
    vals = row.iloc[0].to_dict()
    out = [{"LOS":k, "y":float(pd.to_numeric(vals[c], errors="coerce"))}
           for k,c in zip(list("ABCDEF"), cols) if c in vals and pd.notna(vals[c])]
    return pd.DataFrame(out)

#just the max Q among A-F for one row
def row_qmax_pmin(r):
    cols = ["Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
            "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]
    vals = [pd.to_numeric(r.get(c), errors="coerce") for c in cols]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.nanmax(vals)) if vals else None

# popup code when hovering over arrow in map
def popup_with_info(r, sel_ts):
    fpm_used = pd.to_numeric(r.get("flow_per_min_used"), errors="coerce")
    qmax = row_qmax_pmin(r)
    los  = r.get("LOS_class", "—")
    flow_used_line = f"{fpm_used:.1f} pers/min" if pd.notna(fpm_used) else "—"
    qmax_line = f"{qmax:.1f} pers/min" if qmax is not None else "—"
    return f"""
    <div style='font-family:Inter,system-ui; font-size:13px'>
      <b>{r['SensorBase']}</b><br/>
      <small>{pd.Timestamp(sel_ts):%Y-%m-%d %H:%M}</small><br/>
      direction: {int(r['direction_deg'])}°<br/>
      Flow: <b>{flow_used_line}</b><br/>
      LOS: <b>{los}</b><br/>
      Capacity (Qmax): <b>{qmax_line}</b>
    </div>"""

#add rotated arrow on map
def add_arrow_marker(m, lat, lon, angle_deg, color_hex, hover_html, click_html=None, arrow_size = 22, arrow_symbol = '➔'):
    icon_html = f"""
    <div style="transform: rotate({angle_deg}deg); font-size:{arrow_size}px; line-height:22px;
                color:{color_hex}; text-shadow:0 0 2px #fff;">{arrow_symbol}</div>"""
    marker = folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=icon_html),
        tooltip=folium.Tooltip(hover_html, sticky=True, direction="top", opacity=0.95),
    )
    if click_html:  
        marker.add_child(folium.Popup(click_html, max_width=320))
    marker.add_to(m)

# Determin 'highest' LOS
LOS_RANK = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5}

def los_ge(a, b_min="C"):
    return LOS_RANK.get(str(a), -1) >= LOS_RANK.get(str(b_min), -1)

def build_day_warnings(df_all: pd.DataFrame, sel_day, until_ts=None, los_min="C"):
    daymask = df_all["timestamp"].dt.date == pd.Timestamp(sel_day).date()
    df_day = df_all.loc[daymask].copy()
    if df_day.empty:
        return pd.DataFrame(columns=["time","sensor","directions","hoogste_LOS"])

    # calculate Flow/min 
    if "flow_per_min_smooth" in df_day.columns and df_day["flow_per_min_smooth"].notna().any():
        df_day["flow_per_min_used"] = pd.to_numeric(df_day["flow_per_min_smooth"], errors="coerce")
    elif "flow_per_min" in df_day.columns and df_day["flow_per_min"].notna().any():
        df_day["flow_per_min_used"] = pd.to_numeric(df_day["flow_per_min"], errors="coerce")
    else:
        df_day["flow_per_min_used"] = pd.to_numeric(df_day["count_3min"], errors="coerce")/3.0

    # Qmax columns are numeric 
    for c in ["Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
              "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]:
        if c not in df_day.columns: df_day[c] = pd.NA
        df_day[c] = pd.to_numeric(df_day[c], errors="coerce")

    # Classify LOS per row (per direction)
    df_day["LOS_class"] = df_day.apply(los_from_row, axis=1)

    # Only until chosen timestamp 
    if until_ts is not None:
        df_day = df_day[df_day["timestamp"] <= pd.Timestamp(until_ts)]

    # Filter LOS >= C
    df_hit = df_day[df_day["LOS_class"].apply(lambda x: los_ge(x, los_min))].copy()
    if df_hit.empty:
        return pd.DataFrame(columns=["time","sensor","directions","hoogste_LOS"])

    df_hit = df_hit.sort_values("timestamp")

    out = (df_hit
       .groupby(["SensorBase", "timestamp"], as_index=False)
       .agg(
           direction=("direction_deg", lambda s: ", ".join(sorted({f"{int(d)}°" for d in s.dropna()}))),
           predicted_LOS=("LOS_class", lambda s: max(s, key=lambda x: LOS_RANK.get(x, -1)))
       )
       .rename(columns={"SensorBase": "sensor"}))

    # sort timestamp descending, then sensor ascending
    out = out.sort_values(["timestamp", "sensor"], ascending=[False, True]).reset_index(drop=True)

    # display time
    out["time"] = pd.to_datetime(out["timestamp"]).dt.strftime("%H:%M")

    # only display whats needed
    return out[["time", "sensor", "direction", "predicted_LOS"]]


# Add this function to create forecast warnings
def build_forecast_warnings(forecast_data, sensor_base, df_all, los_min="C"):
    """
    Build warnings based on forecasted data for the next 2 hours
    """
    if not forecast_data:
        return pd.DataFrame(columns=["time","sensor","directions","predicted_LOS"])
    
    # Get Qmax values for this sensor
    sensor_row = df_all[df_all["SensorBase"] == sensor_base].iloc[0] if not df_all[df_all["SensorBase"] == sensor_base].empty else None
    if sensor_row is None:
        return pd.DataFrame(columns=["time","sensor","directions","predicted_LOS"])
    
    # Extract Qmax values
    qmax_A = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_A"), errors="coerce")
    qmax_B = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_B"), errors="coerce")
    qmax_C = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_C"), errors="coerce")
    qmax_D = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_D"), errors="coerce")
    qmax_E = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_E"), errors="coerce")
    qmax_F = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_F"), errors="coerce")
    
    warnings = []
    
    # Process each direction's forecast
    for direction, forecast_series in forecast_data.items():
        for timestamp, flow in forecast_series.items():
            # Calculate LOS for this forecast point
            if pd.isna(flow) or pd.isna(qmax_A):
                continue
                
            if flow <= qmax_A: 
                los = "A"
            elif flow <= qmax_B: 
                los = "B"
            elif flow <= qmax_C: 
                los = "C"
            elif flow <= qmax_D: 
                los = "D"
            elif flow <= qmax_E: 
                los = "E"
            else: 
                los = "F"
            
            # Check if this LOS meets our warning threshold
            if los_ge(los, los_min):
                warnings.append({
                    "time": timestamp.strftime("%H:%M"),
                    "sensor": sensor_base,
                    "directions": f"{direction}°",
                    "predicted_LOS": los,
                    "type": "forecast"
                })
    
    return pd.DataFrame(warnings)

def build_forecast_warnings_all(forecast_data_all, df_all, los_min="C"):
    """Build warnings based on forecasted data for all sensors"""
    if not forecast_data_all:
        return pd.DataFrame(columns=["time","sensor","directions","predicted_LOS"])
    
    warnings = []
    
    for sensor_base, forecast_data in forecast_data_all.items():
        # Get Qmax values for this sensor
        sensor_rows = df_all[df_all["SensorBase"] == sensor_base]
        if sensor_rows.empty:
            continue
            
        sensor_row = sensor_rows.iloc[0]
        
        # Extract Qmax values
        qmax_A = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_A"), errors="coerce")
        qmax_B = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_B"), errors="coerce")
        qmax_C = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_C"), errors="coerce")
        qmax_D = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_D"), errors="coerce")
        qmax_E = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_E"), errors="coerce")
        qmax_F = pd.to_numeric(sensor_row.get("Q_max_pers_per_min_LOS_F"), errors="coerce")
        
        # Process each direction's forecast
        for direction, forecast_series in forecast_data.items():
            for timestamp, flow in forecast_series.items():
                # Calculate LOS for this forecast point
                if pd.isna(flow) or pd.isna(qmax_A):
                    continue
                    
                if flow <= qmax_A: 
                    los = "A"
                elif flow <= qmax_B: 
                    los = "B"
                elif flow <= qmax_C: 
                    los = "C"
                elif flow <= qmax_D: 
                    los = "D"
                elif flow <= qmax_E: 
                    los = "E"
                else: 
                    los = "F"
                
                # Check if this LOS meets our warning threshold
                if los_ge(los, los_min):
                    warnings.append({
                        "time": timestamp.strftime("%H:%M"),
                        "sensor": sensor_base,
                        "directions": f"{direction}°",
                        "predicted_LOS": los,
                        "type": "forecast"
                    })
    
    return pd.DataFrame(warnings)

    
#DISPLAY:
###############################################################


#side bar
#############
def side_bar(df_all: pd.DataFrame):
    with st.sidebar:
        st.markdown('## ⚙️ Settings')
        st.markdown('### Day Overview per sensor')
        sensors = sorted(df_all["SensorBase"].dropna().unique().tolist())
        sel_sensor = st.selectbox("Sensor", sensors, index=0, key="sensor_day_overview")

        # all available days for this sensor 
        dates = (df_all.loc[df_all["SensorBase"] == sel_sensor, "timestamp"]
                    .dropna()
                    .dt.date
                    .unique())
        dates = sorted(dates.tolist())

        # dropdown: day name + date 
        sel_date = st.selectbox(
            "Day",
            options=dates if dates else [pd.Timestamp.today().date()],
            index=len(dates) - 1 if dates else 0,
            format_func=lambda d: pd.Timestamp(d).day_name(locale="en_US") + " " + pd.Timestamp(d).strftime("%d-%m-%Y"),  # gebruikt je bestaande helper
            key=f"day_sensor_select_{sel_sensor}"
        )
    return sel_sensor, sel_date  

# Side bar for map overview
def side_bar_map(df_all: pd.DataFrame):
        with st.sidebar:
            st.markdown('## ⚙️ Settings')
            st.markdown('### Map display')
            arrow_size = st.slider("Arrow size", min_value = 5, max_value = 100, value = 22)

            arrow_types = '➔', '➤', '⟼', '➲', '⇢', '➣'
            arrow_symbol = st.selectbox("Arrow type", arrow_types)

        return arrow_size, arrow_symbol 


#raw data tab
###########################
def raw_data(df_all: pd.DataFrame):
    sel_sensor, sel_date = side_bar(df_all)

    mask = (df_all["SensorBase"]==sel_sensor) & (df_all["timestamp"].dt.date==pd.to_datetime(sel_date).date())
    want = ["timestamp","SensorBase","direction_deg","count_3min","flow_per_min",
            "Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
            "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]
    have = [c for c in want if c in df_all.columns]
    df_day = df_all.loc[mask, have].copy()

    if df_day.empty:
        st.info("No data for this sensor and day.")
        return  
    
    if "flow_per_min_smooth" in df_day.columns and df_day["flow_per_min_smooth"].notna().any():
        df_day["flow_min_used"] = pd.to_numeric(df_day["flow_per_min_smooth"], errors="coerce")
    elif "flow_per_min" in df_day.columns and df_day["flow_per_min"].notna().any():
        df_day["flow_min_used"] = pd.to_numeric(df_day["flow_per_min"], errors="coerce")
    else:
        df_day["flow_min_used"] = pd.to_numeric(df_day["count_3min"], errors="coerce")/3.0

    df_day.columns = ['Time', 'Sensor name', 'Direction', 'ppl per 3 min', 'LOS A limit', 'LOS B limit', 'LOS C limit',
                      'LOS D limit', 'LOS E limit', 'LOS F limit', 'Flow per min']

    st.write('## The raw data of the sensors in the SAIL event')
    st.dataframe(df_day, height= 700)


#sensor data per day tab
#########################
def render_sensor_day_overview(df_all: pd.DataFrame, model_data=None):
    sel_sensor, sel_date = side_bar(df_all)

    mask = (df_all["SensorBase"]==sel_sensor) & (df_all["timestamp"].dt.date==pd.to_datetime(sel_date).date())
    want = ["timestamp","SensorBase","direction_deg","count_3min","flow_per_min",
            "Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
            "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]
    have = [c for c in want if c in df_all.columns]
    df_day = df_all.loc[mask, have].copy()

    if df_day.empty:
        st.info("No data for this sensor and day.")
        return

    if "flow_per_min_smooth" in df_day.columns and df_day["flow_per_min_smooth"].notna().any():
        df_day["flow_min_used"] = pd.to_numeric(df_day["flow_per_min_smooth"], errors="coerce")
    elif "flow_per_min" in df_day.columns and df_day["flow_per_min"].notna().any():
        df_day["flow_min_used"] = pd.to_numeric(df_day["flow_per_min"], errors="coerce")
    else:
        df_day["flow_min_used"] = pd.to_numeric(df_day["count_3min"], errors="coerce")/3.0

    df_day["direction"] = df_day.get("direction_deg", np.nan).astype("Int64").astype(str) + "°"
    y_max = float(df_day["flow_min_used"].max() or 0) * 1.1 or 1.0

    flow_lines = alt.Chart(df_day).mark_line().encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("flow_min_used:Q", title="Flow (per min)"),
        color=alt.Color("direction:N", title="Direction"),
        tooltip=[alt.Tooltip("timestamp:T", title="Time"),
                 alt.Tooltip("direction:N", title="Direction"),
                 alt.Tooltip("flow_min_used:Q", title="Flow/min", format=".1f")]
    ).properties(height=280)

    # LOS-lines from 1 row (constant per sensor)
    rules = []
    th = df_day.iloc[0]
    for label, col in zip(list("ABCDEF"),
                          ["Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
                           "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]):
        val = pd.to_numeric(th.get(col), errors="coerce")
        if pd.notna(val) and float(val) <= y_max:
            rules.append(
                alt.Chart(pd.DataFrame({"y":[float(val)], "LOS":[label]}))
                   .mark_rule(strokeDash=[6,4])
                   .encode(y="y:Q",
                           color=alt.Color("LOS:N", scale=alt.Scale(domain=list("ABCDEF"), range=LOS_COLOR_LIST),
                                           title="LOS-lijn"))
            )

    chart = alt.layer(flow_lines, *rules).resolve_scale(color='independent')

    # margin for readable x-axis
    chart = chart.configure_view(
        stroke=None
    ).configure_axisX(
        labelPadding=10,
        labelAngle=0
    ).configure(                      
        padding={"bottom": 20}
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Sensor: **{sel_sensor}**, Date: **{sel_date}**. Los boundaries only appear when in y-reach.")


#Map, mini-trend and warning tab (home sceen)
#############################################
def render_map_and_trend(df_all: pd.DataFrame, model_data=None):
    arrow_size, arrow_symbol = side_bar_map(df_all)

    #  Day + slider 
    ts_values = sorted(pd.to_datetime(df_all["timestamp"].unique()))
    all_dates = sorted(pd.to_datetime(df_all["timestamp"]).dt.date.unique().tolist())
    if not all_dates:
        st.warning("No timestamps available"); return

    # day choice 
    sel_day = st.selectbox(
        "Day",
        options=all_dates,
        index=len(all_dates)-1,
        format_func=lambda d: pd.Timestamp(d).day_name(locale="en_US") + " " + pd.Timestamp(d).strftime("%d-%m-%Y"),
        key="day_map_select"
    )

    # all timestamps in chosen day
    ts_day = [t for t in ts_values if t.date() == sel_day]
    if not ts_day:  # safety
        st.warning("Geen metingen op deze dag."); return
    ts_min_day, ts_max_day = ts_day[0], ts_day[-1]

    # only show HH:MM
    eff_ts = pd.Timestamp(
        st.slider(
            "Slide for choosing a timestamp",
            min_value=ts_min_day.to_pydatetime(),
            max_value=ts_max_day.to_pydatetime(),
            value=ts_max_day.to_pydatetime(),
            step=pd.Timedelta(minutes=1),
            format="HH:mm",
            key=f"slider_{sel_day}"
        )
    )

    # filter with base eff_ts
    # Exact match (min), or closest
    df_time = df_all[df_all["timestamp"] == eff_ts].copy()
    if df_time.empty:
        nearest = min(ts_day, key=lambda x: abs(x - eff_ts))
        eff_ts = pd.Timestamp(nearest)
        df_time = df_all[df_all["timestamp"] == eff_ts].copy()

    # calculate Flow/min + LOS per row (for map colours)
    if "flow_per_min_smooth" in df_time.columns and df_time["flow_per_min_smooth"].notna().any():
        df_time["flow_per_min_used"] = pd.to_numeric(df_time["flow_per_min_smooth"], errors="coerce")
    else:
        df_time["flow_per_min_used"] = pd.to_numeric(df_time["count_3min"], errors="coerce") / 3.0

    for c in ["Q_max_pers_per_min_LOS_A","Q_max_pers_per_min_LOS_B","Q_max_pers_per_min_LOS_C",
              "Q_max_pers_per_min_LOS_D","Q_max_pers_per_min_LOS_E","Q_max_pers_per_min_LOS_F"]:
        if c not in df_time.columns: df_time[c] = pd.NA
        df_time[c] = pd.to_numeric(df_time[c], errors="coerce")

    df_time["LOS_class"] = df_time.apply(los_from_row, axis=1)

    # focus sensor
    if "focus_sensor" not in st.session_state and not df_time.empty:
        st.session_state.focus_sensor = df_time["SensorBase"].iloc[0]
    focus = st.session_state.get("focus_sensor")

    # layout: map left links, mini-trend and warnings tab right
    left, right = st.columns([7, 5], vertical_alignment="top")
    #left side of the screen
    with left:
        # map
        MIN_ZOOM = 12
        CENTER_AMS = [52.3728, 4.8936]

        m = folium.Map(location=CENTER_AMS, zoom_start=MIN_ZOOM, tiles=None, zoom_control=True)
        folium.TileLayer(
            tiles="CartoDB Positron", name="Base", attr="© Carto",
            min_zoom=MIN_ZOOM, max_zoom=19, control=False
        ).add_to(m)

        # markers
        df_time = df_time.dropna(subset=["latitude","longitude"]).copy()

        # recalculate LOS_class to be sure
        if "LOS_class" not in df_time.columns or df_time["LOS_class"].isna().all():
            if "flow_per_min_smooth" in df_time.columns:
                df_time["flow_per_min_used"] = pd.to_numeric(df_time["flow_per_min_smooth"], errors="coerce")
            else:
                df_time["flow_per_min_used"] = pd.to_numeric(df_time["count_3min"], errors="coerce")/3.0
            df_time["LOS_class"] = df_time.apply(los_from_row, axis=1)

        LOS_TO_COLOR = LOS_COLORS  
        for _, r in df_time.iterrows():
            los_val = r.get("LOS_class")
            color_hex = LOS_TO_COLOR.get(los_val, "#9E9E9E")
            detail_html = popup_with_info(r, eff_ts)
            add_arrow_marker(
                m,
                r["latitude"], r["longitude"],
                int(r.get("direction_deg") or 0),
                color_hex,
                hover_html=detail_html,
                click_html=detail_html,
                arrow_size = arrow_size,
                arrow_symbol = arrow_symbol
            )



        # LOS-legend as overlay in map
        legend_html = f"""
        <div style="
            position: absolute; bottom: 14px; left: 14px; z-index: 9999;
            background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2); font-size: 12px;">
            <b>LOS</b><br>
            {''.join([
                f'<div><span style="display:inline-block;width:10px;height:10px;'
                f'background:{LOS_COLORS[k]};border:1px solid #888;margin-right:6px;"></span>{k}</div>'
                for k in "ABCDEF"
            ])}
        </div>"""

        m.get_root().html.add_child(folium.Element(legend_html))

        # render map
        st_map = st_folium(
            m,
            height=380,
            use_container_width=True,
            returned_objects=["last_object_clicked","last_object_clicked_popup"],
            key=f"map_{eff_ts.strftime('%Y%m%d_%H%M')}"  # forceer rerender per tijdstip
        )

        # caption below map
        st.caption("Hover over a sensor to view live flow and LOS details, or click a sensor to display its last 60 minutes in the graph on the right.")

        # click: focus_sensor update (works for pop-up and only-click) 
        clicked_name = None

        # try to remove popup/tooltip html out of sensortitle 
        popup_html = st_map.get("last_object_clicked_popup")
        if popup_html:
            m_ = re.search(r"<b>\s*([^<]+?)\s*</b>", str(popup_html))
            if m_:
                clicked_name = normalize_name(m_.group(1))

        # if not: use lat/lon of click and find closest marker of timestamp
        if not clicked_name and st_map.get("last_object_clicked") and not df_time.empty:
            lat = st_map["last_object_clicked"].get("lat")
            lon = st_map["last_object_clicked"].get("lng")
            if lat is not None and lon is not None:
                c = np.cos(np.radians(lat))
                dx = (df_time["longitude"] - lon) * c
                dy = (df_time["latitude"]  - lat)
                idx = (dx*dx + dy*dy).idxmin()
                clicked_name = normalize_name(str(df_time.loc[idx, "SensorBase"]))

        # 3) new focus and rerun
        if clicked_name and clicked_name != normalize_name(st.session_state.get("focus_sensor")):
            st.session_state["focus_sensor"] = clicked_name
            st.rerun()
    
    #right side of the screen
    #right side of the screen
    with right:
        # Mini-trend (compact) 
        series = get_last_minutes_by_direction(df_all, focus, eff_ts, minutes=60)
        
        # Get forecasts for ALL sensors if model is available
        forecast_data_all = {}
        if model_data:
            model, scaler, feature_cols, sensor_cols = model_data
            try:
                forecast_data_all = get_all_forecasts(
                    df_all, model, scaler, feature_cols, sensor_cols, eff_ts
                )
            except Exception as e:
                st.error(f"Forecast error: {e}")
        
        # Get forecast for focused sensor for the mini-trend chart
        focus_forecast_data = forecast_data_all.get(focus, {}) if focus else {}
        
        if series.empty and not focus_forecast_data:
            st.subheader(f"Mini-trend – {focus or '—'}")
            st.info("No data in the last 60 minutes.")
        else:
            series["direction"] = series["direction_deg"].astype("Int64").astype(str) + "°"
            
            # Prepare data for chart - combine historical and forecast
            chart_data = series.copy()
            chart_data['type'] = 'historical'
            
            # Add forecast data if available
            forecast_chart_data = []
            if focus_forecast_data:
                for direction, forecast_series in focus_forecast_data.items():
                    forecast_df = forecast_series.reset_index()
                    forecast_df.columns = ['timestamp', 'flow_per_min_used']
                    forecast_df['direction'] = direction + "°"
                    forecast_df['type'] = 'forecast'
                    forecast_chart_data.append(forecast_df)
                
                if forecast_chart_data:
                    forecast_combined = pd.concat(forecast_chart_data, ignore_index=True)
                    chart_data = pd.concat([chart_data, forecast_combined], ignore_index=True)

            # scale + padding
            y_min, y_max = float(chart_data["flow_per_min_used"].min()), float(chart_data["flow_per_min_used"].max())
            pad = max(6.0, 0.12 * (y_max - y_min if y_max > y_min else 1.0))
            ydomain = [max(0.0, y_min - pad), y_max + pad]

            los_df = get_los_lines(df_all, focus)
            los_in = los_df[(los_df["y"] >= ydomain[0]) & (los_df["y"] <= ydomain[1])] if not los_df.empty else los_df

            # Create separate charts for historical and forecast data
            historical_data = chart_data[chart_data['type'] == 'historical']
            forecast_data_chart = chart_data[chart_data['type'] == 'forecast']
            
            historical_lines = alt.Chart(historical_data).mark_line().encode(
                x=alt.X("timestamp:T", axis=alt.Axis(title=None, format="%H:%M")),
                y=alt.Y("flow_per_min_used:Q",
                        axis=alt.Axis(title="pers/min", labelFlush=False),
                        scale=alt.Scale(domain=ydomain, nice=False)),
                color=alt.Color("direction:N", title="Direction")
            ).properties(height=190)  
            
            forecast_lines = alt.Chart(forecast_data_chart).mark_line(strokeDash=[5,3]).encode(
                x=alt.X("timestamp:T", axis=alt.Axis(title=None, format="%H:%M")),
                y=alt.Y("flow_per_min_used:Q",
                        axis=alt.Axis(title="pers/min", labelFlush=False),
                        scale=alt.Scale(domain=ydomain, nice=False)),
                color=alt.Color("direction:N", title="Direction")
            ).properties(height=190)

            chart = historical_lines
            if not forecast_data_chart.empty:
                chart = historical_lines + forecast_lines

            if not los_in.empty:
                rules = alt.Chart(los_in).mark_rule(strokeDash=[6,4], opacity=0.9).encode(
                    y="y:Q",
                    color=alt.Color("LOS:N", title="LOS-Colors",
                                    scale=alt.Scale(domain=list("ABCDEF"),
                                                    range=[LOS_COLORS[k] for k in "ABCDEF"]))
                )
                chart = (rules + chart).resolve_scale(color='independent')

            chart = chart.configure_axisX(
                labelAngle=0,        # horizontal
                labelPadding=10,     # extra room below labels
                labelOverlap=False   # force all labels
            ).configure_view(
                stroke=None
            )

            st.subheader(f" Sensor {normalize_name(focus)}: last 60 min + 2h forecast")
            st.altair_chart(chart, use_container_width=True)
            st.caption("Solid lines: historical data. Dashed lines: 2-hour forecast. LOS boundary lines only appear when in y-reach.")

        # Warnings section - both historical and forecast
        st.markdown("### ⚠️ Warnings")
        
        # Historical warnings (up to current time)
        historical_warnings = build_day_warnings(df_all, sel_day, until_ts=eff_ts, los_min="C")
        
        # Forecast warnings (next 2 hours) for ALL sensors
        forecast_warnings = pd.DataFrame()
        if model_data and forecast_data_all:
            forecast_warnings = build_forecast_warnings_all(forecast_data_all, df_all, los_min="C")
            if not forecast_warnings.empty:
                forecast_warnings["type"] = "forecast"
        
        # Combine and display warnings
        if historical_warnings.empty and forecast_warnings.empty:
            st.success("No warnings (historical or forecasted) up until this time.")
        else:
            # Create tabs for historical vs forecast warnings
            tab1, tab2 = st.tabs(["Historical Warnings", "Forecast Warnings"])
            
            with tab1:
                if not historical_warnings.empty:
                    historical_warnings = historical_warnings.sort_values("time", ascending=False).reset_index(drop=True)
                    
                    def _style_los_historical(col):
                        return [f"color: {LOS_COLORS.get(str(v), '#666')}; font-weight: 700" for v in col]
                    
                    styled_historical = historical_warnings.style.apply(_style_los_historical, subset=["predicted_LOS"])
                    
                    st.dataframe(
                        styled_historical,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("Shows every minute a sensor reached LOS ≥ C up to the current time (newest first).")
                else:
                    st.info("No historical warnings up to current time.")
            
            with tab2:
                if not forecast_warnings.empty:
                    forecast_warnings = forecast_warnings.sort_values("time", ascending=True).reset_index(drop=True)
                    
                    def _style_los_forecast(col):
                        return [f"color: {LOS_COLORS.get(str(v), '#666')}; font-weight: 700; font-style: italic" for v in col]
                    
                    styled_forecast = forecast_warnings.style.apply(_style_los_forecast, subset=["predicted_LOS"])
                    
                    st.dataframe(
                        styled_forecast,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("Predicted LOS ≥ C for the next 2 hours across all sensors (chronological order).")
                else:
                    if model_data:
                        st.success("No forecasted warnings for any sensor in the next 2 hours.")
                    else:
                        st.info("Load forecasting model to see forecast warnings for all sensors.")
    
#DATA PIPELINE
flow_long = load_flow_long(FLOW_CSV)
locations = load_locations(LOC_CSV)
qmax_af   = load_qmax(QMAX_CSV)

flow_long["timestamp"] = pd.to_datetime(flow_long["timestamp"], errors="coerce")
df_all = build_df_all(flow_long, locations, qmax_af)

# Load forecasting model
with st.sidebar:
    st.markdown("## Forecasting Model")
    if st.button("Load Forecasting Model"):
        with st.spinner("Loading forecasting model..."):
            model_data = load_forecast_model()
            if model_data[0] is not None:
                st.session_state.model_data = model_data
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")

# Check if model is loaded
model_data = st.session_state.get('model_data', None)

#SIDEBAR ROUTER
with st.sidebar:
    st.markdown("## Display")
    view = st.radio("Choose overview:",
                    ["Map", "Day overview per Sensor", "Raw sensor data"], index=0)

#RENDER
if view == "Map":
    render_map_and_trend(df_all, model_data)
elif view == "Raw sensor data":
    raw_data(df_all)
else:
    render_sensor_day_overview(df_all, model_data)