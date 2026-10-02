import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Road Safety Explorer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef6ff 0%,
        #f8fbff 50%,
        #ffffff 100%
    );
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1450px;
}

/* Main title */
h1 {
    color: #12355b !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    color: #1d4e89 !important;
    font-weight: 650 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.35rem !important;
}

h4 {
    color: #245b8f !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.25rem !important;
}

p, label, .stMarkdown {
    color: #243447;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.90);
    border: 1px solid #d8e6f3;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 3px 12px rgba(30, 70, 100, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #496579 !important;
}

[data-testid="stMetricValue"] {
    color: #12355b !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #cbddec;
}

/* Sliders */
div[data-testid="stSlider"] {
    background: rgba(255, 255, 255, 0.70);
    padding: 10px 15px;
    border-radius: 10px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #d7e5f2;
}

/* Hide footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = Path("Road Accident Data.xlsx")
MODEL_PATH = Path("severity_rf_model.pkl")


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = [
    "Day_of_Week",
    "Light_Conditions",
    "Local_Authority_(District)",
    "Road_Surface_Conditions",
    "Road_Type",
    "Speed_limit",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type",
    "Month",
    "Hour"
]


# =========================================================
# CREATE HOUR COLUMN
# =========================================================

def create_hour_column(df):

    if "Hour" in df.columns:
        return df

    if "Time" not in df.columns:
        df["Hour"] = 12
        return df

    def extract_hour(value):

        try:

            if pd.isna(value):
                return 12

            # Excel fractional time
            if isinstance(value, (int, float)):
                if 0 <= value < 1:
                    return int(value * 24)
                return int(value)

            # Timestamp
            if isinstance(value, pd.Timestamp):
                return value.hour

            # Python time
            if hasattr(value, "hour"):
                return value.hour

            # String time
            parsed = pd.to_datetime(
                str(value),
                errors="coerce"
            )

            if not pd.isna(parsed):
                return parsed.hour

        except:
            pass

        return 12

    df["Hour"] = df["Time"].apply(extract_hour)

    df["Hour"] = (
        pd.to_numeric(
            df["Hour"],
            errors="coerce"
        )
        .fillna(12)
        .astype(int)
        .clip(0, 23)
    )

    return df


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel(DATA_PATH)

    # Correct severity typo
    if "Accident_Severity" in df.columns:

        df["Accident_Severity"] = (
            df["Accident_Severity"]
            .replace({"Fetal": "Fatal"})
        )

    # Create Hour
    df = create_hour_column(df)

    # Create Day and Month
    if "Accident Date" in df.columns:

        date_col = pd.to_datetime(
            df["Accident Date"],
            errors="coerce"
        )

        if "Day_of_Week" not in df.columns:
            df["Day_of_Week"] = date_col.dt.day_name()

        if "Month" not in df.columns:
            df["Month"] = date_col.dt.month_name()

    # Fill missing categorical values
    categorical_columns = [
        "Carriageway_Hazards",
        "Road_Surface_Conditions",
        "Road_Type",
        "Weather_Conditions",
        "Light_Conditions",
        "Junction_Control",
        "Junction_Detail",
        "Urban_or_Rural_Area",
        "Vehicle_Type",
        "Local_Authority_(District)"
    ]

    for col in categorical_columns:

        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Speed limit numeric
    if "Speed_limit" in df.columns:

        df["Speed_limit"] = pd.to_numeric(
            df["Speed_limit"],
            errors="coerce"
        )

    return df


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# =========================================================
# LOAD FILES
# =========================================================

try:

    df = load_data()
    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the dataset or trained model."
    )

    st.code(str(e))
    st.stop()


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Accident_Severity",
    "Local_Authority_(District)",
    "Hour",
    "Day_of_Week",
    "Month",
    "Speed_limit"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🚦 Road Safety Explorer")

st.caption(
    "Understand historical accident patterns for a location "
    "and explore conditions for a planned journey."
)


# =========================================================
# LOCATION SELECTION
# =========================================================

location_col, empty_col = st.columns([1.4, 4])

with location_col:

    districts = sorted(
        df["Local_Authority_(District)"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_district = st.selectbox(
        "Select a location",
        districts
    )


# =========================================================
# LOCATION DATA
# =========================================================

location_df = df[
    df["Local_Authority_(District)"].astype(str)
    == selected_district
].copy()


# =========================================================
# HISTORICAL ACCIDENT OVERVIEW
# =========================================================

st.markdown("### Historical accident overview")

total_records = len(location_df)

peak_hour = (
    location_df["Hour"]
    .value_counts()
    .idxmax()
)

peak_day = (
    location_df["Day_of_Week"]
    .value_counts()
    .idxmax()
)

common_severity = (
    location_df["Accident_Severity"]
    .value_counts()
    .idxmax()
)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Recorded accident records",
        f"{total_records:,}"
    )

with m2:
    st.metric(
        "Peak recorded hour",
        f"{int(peak_hour):02d}:00"
    )

with m3:
    st.metric(
        "Peak recorded day",
        peak_day
    )

with m4:
    st.metric(
        "Most common severity",
        common_severity
    )


# =========================================================
# HISTORICAL CHARTS
# =========================================================

chart_col1, chart_col2 = st.columns(2)


# =========================================================
# HISTORICAL ACCIDENTS BY HOUR
# =========================================================

with chart_col1:

    st.markdown("#### Historical accidents by hour")

    hourly = (
        location_df["Hour"]
        .value_counts()
        .sort_index()
        .reindex(range(24), fill_value=0)
    )

    hourly_df = pd.DataFrame({
        "Hour": [
            f"{h:02d}:00"
            for h in range(24)
        ],
        "Accident records": hourly.values
    })

    st.bar_chart(
        hourly_df.set_index("Hour"),
        height=250
    )


# =========================================================
# SEVERITY BY SPEED LIMIT
# =========================================================

with chart_col2:

    st.markdown("#### Severity by speed limit")

    speed_severity = pd.crosstab(
        location_df["Speed_limit"],
        location_df["Accident_Severity"],
        normalize="index"
    ) * 100

    speed_severity = (
        speed_severity
        .reindex(
            columns=[
                "Slight",
                "Serious",
                "Fatal"
            ],
            fill_value=0
        )
        .sort_index()
    )

    st.line_chart(
        speed_severity,
        height=250
    )

    st.caption(
        "Percentage distribution of severity within each speed-limit category."
    )


# =========================================================
# SEVERITY DISTRIBUTION + INSIGHTS
# =========================================================

severity_col, insight_col = st.columns([1, 2])


# =========================================================
# SEVERITY DISTRIBUTION
# =========================================================

with severity_col:

    st.markdown("#### Severity distribution")

    severity_counts = (
        location_df["Accident_Severity"]
        .value_counts()
        .reindex(
            ["Slight", "Serious", "Fatal"],
            fill_value=0
        )
    )

    st.bar_chart(
        severity_counts,
        height=180
    )


# =========================================================
# KEY HISTORICAL INSIGHTS
# =========================================================

with insight_col:

    st.markdown("#### Key historical insights")

    # Night
    night_df = location_df[
        location_df["Hour"].isin(
            [0, 1, 2, 3, 4, 5]
        )
    ]

    if len(night_df) > 0:

        night_fatal_rate = (
            (
                night_df["Accident_Severity"]
                == "Fatal"
            ).mean()
            * 100
        )

        st.markdown(
            f"• During **12 AM–5 AM**, Fatal accidents "
            f"represented **{night_fatal_rate:.2f}%** "
            f"of recorded accidents."
        )

    # Rural / Urban
    if "Urban_or_Rural_Area" in location_df.columns:

        rural_df = location_df[
            location_df["Urban_or_Rural_Area"]
            .astype(str)
            .str.lower()
            .str.contains("rural")
        ]

        urban_df = location_df[
            location_df["Urban_or_Rural_Area"]
            .astype(str)
            .str.lower()
            .str.contains("urban")
        ]

        if len(rural_df) > 0 and len(urban_df) > 0:

            rural_rate = (
                rural_df["Accident_Severity"]
                .isin(["Serious", "Fatal"])
                .mean()
                * 100
            )

            urban_rate = (
                urban_df["Accident_Severity"]
                .isin(["Serious", "Fatal"])
                .mean()
                * 100
            )

            st.markdown(
                f"• Serious/Fatal accidents represented "
                f"**{rural_rate:.1f}%** of rural records "
                f"versus **{urban_rate:.1f}%** of urban records."
            )

    # Speed
    high_speed_df = location_df[
        location_df["Speed_limit"] >= 50
    ]

    if len(high_speed_df) > 0:

        high_speed_rate = (
            high_speed_df["Accident_Severity"]
            .isin(["Serious", "Fatal"])
            .mean()
            * 100
        )

        st.markdown(
            f"• At speed limits of **50 mph or above**, "
            f"Serious/Fatal accidents represented "
            f"**{high_speed_rate:.1f}%** of records."
        )


# =========================================================
# JOURNEY CONDITIONS
# =========================================================

st.markdown("---")

st.markdown("### Check your journey conditions")


# =========================================================
# FIRST INPUT ROW
# =========================================================

input1, input2, input3, input4, input5, input6 = st.columns(6)


# Time
with input1:

    journey_hour = st.slider(
        "Journey time",
        0,
        23,
        12,
        format="%d:00"
    )


# Day
with input2:

    day_options = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    journey_day = st.selectbox(
        "Day",
        day_options
    )


# Month
with input3:

    month_options = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    journey_month = st.selectbox(
        "Month",
        month_options
    )


# Weather
with input4:

    weather_options = sorted(
        df["Weather_Conditions"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_weather = st.selectbox(
        "Weather",
        weather_options
    )


# Light
with input5:

    light_options = sorted(
        df["Light_Conditions"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_light = st.selectbox(
        "Light",
        light_options
    )


# Road type
with input6:

    road_options = sorted(
        df["Road_Type"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_road = st.selectbox(
        "Road type",
        road_options
    )


# =========================================================
# SECOND INPUT ROW
# =========================================================

input7, input8, input9, input10, input11 = st.columns(5)


# Road surface
with input7:

    surface_options = sorted(
        df["Road_Surface_Conditions"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_surface = st.selectbox(
        "Road surface",
        surface_options
    )


# Speed
with input8:

    speed_options = sorted(
        df["Speed_limit"]
        .dropna()
        .unique()
    )

    journey_speed = st.selectbox(
        "Speed limit",
        speed_options
    )


# Area
with input9:

    urban_options = sorted(
        df["Urban_or_Rural_Area"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_area = st.selectbox(
        "Area",
        urban_options
    )


# Vehicle
with input10:

    vehicle_options = sorted(
        df["Vehicle_Type"]
        .dropna()
        .astype(str)
        .unique()
    )

    journey_vehicle = st.selectbox(
        "Vehicle type",
        vehicle_options
    )


# District
with input11:

    journey_district = st.selectbox(
        "District",
        districts,
        index=(
            districts.index(selected_district)
            if selected_district in districts
            else 0
        )
    )


# =========================================================
# JOURNEY SUMMARY + HISTORICAL CONTEXT
# =========================================================

summary_col, context_col = st.columns(2)


# =========================================================
# JOURNEY SUMMARY
# =========================================================

with summary_col:

    st.markdown("#### Journey summary")

    st.info(
        f"📍 **{journey_district}**  |  "
        f"🕐 **{journey_hour:02d}:00**  |  "
        f"📅 **{journey_day}, {journey_month}**  |  "
        f"🌦️ **{journey_weather}**  |  "
        f"💡 **{journey_light}**  |  "
        f"🛣️ **{journey_road}**"
    )


# =========================================================
# HISTORICAL CONTEXT
# =========================================================

with context_col:

    st.markdown("#### Historical context")

    journey_location_df = df[
        df["Local_Authority_(District)"]
        .astype(str)
        == str(journey_district)
    ]

    if len(journey_location_df) > 0:

        location_peak_hour = (
            journey_location_df["Hour"]
            .value_counts()
            .idxmax()
        )

        if int(location_peak_hour) == int(journey_hour):

            st.info(
                f"Accidents have historically been recorded "
                f"most often around "
                f"**{int(location_peak_hour):02d}:00** "
                f"in this district, which matches your "
                f"selected journey time."
            )

        else:

            st.info(
                f"Accidents have historically been recorded "
                f"most often around "
                f"**{int(location_peak_hour):02d}:00** "
                f"in this district. Your selected journey "
                f"time is **{int(journey_hour):02d}:00**."
            )


# =========================================================
# MODEL PREDICTION
# =========================================================

st.markdown(
    "#### If an accident occurs, the predicted severity would be:"
)


# =========================================================
# MODEL INPUT
# =========================================================

journey_data = pd.DataFrame([{

    "Day_of_Week": journey_day,

    "Light_Conditions": journey_light,

    "Local_Authority_(District)": journey_district,

    "Road_Surface_Conditions": journey_surface,

    "Road_Type": journey_road,

    "Speed_limit": journey_speed,

    "Urban_or_Rural_Area": journey_area,

    "Weather_Conditions": journey_weather,

    "Vehicle_Type": journey_vehicle,

    "Month": journey_month,

    "Hour": journey_hour

}])


# =========================================================
# PREDICTION
# =========================================================

try:

    prediction = model.predict(
        journey_data[FEATURES]
    )[0]

except Exception as e:

    prediction = "Unavailable"


# =========================================================
# DISPLAY PREDICTION
# =========================================================

pred_col1, pred_col2 = st.columns([1, 3])


with pred_col1:

    if prediction == "Fatal":

        st.error("🔴 Fatal")

    elif prediction == "Serious":

        st.warning("🟠 Serious")

    elif prediction == "Slight":

        st.success("🟢 Slight")

    else:

        st.error(
            "Prediction unavailable"
        )


with pred_col2:

    st.info(
        "Based on the selected journey conditions and "
        "patterns learned from historical accident records, "
        f"the model predicts **{prediction}** as the most "
        "likely accident severity if an accident were to occur."
    )

    st.caption(
        "This model predicts accident severity conditional "
        "on an accident; it does not predict whether an "
        "accident will happen."
    )


# =========================================================
# JOURNEY PRECAUTIONS
# =========================================================

st.markdown("#### Journey precautions")

p1, p2, p3 = st.columns(3)

with p1:

    st.markdown(
        "🚗 **Follow the posted speed limit** and maintain "
        "a safe following distance."
    )

with p2:

    st.markdown(
        "💡 **Use appropriate lighting** and remain alert "
        "when visibility is reduced."
    )

with p3:

    st.markdown(
        "🌧️ **Adjust driving to weather and road conditions**, "
        "especially on wet or slippery roads."
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    "⚠️ Educational project: This dashboard uses historical "
    "UK road accident records for analysis and learning. "
    "Historical patterns do not guarantee future outcomes. "
    "The severity prediction is not a substitute for "
    "professional road-safety guidance."
)