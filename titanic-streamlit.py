import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Titanic Dataset Explorer", layout="wide")

# ---- Load & tidy -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    # Friendlier Embarked names
    df.loc[df["Embarked"] == "C", "Embarked"] = "Cherbourg"
    df.loc[df["Embarked"] == "Q", "Embarked"] = "Queenstown"
    df.loc[df["Embarked"] == "S", "Embarked"] = "Southampton"

    # Ensure consistent dtypes
    # (Survived is 0/1; keep as int, but some CSVs may read as float)
    if df["Survived"].dtype != int:
        df["Survived"] = df["Survived"].astype(int)

    return df

df = load_data()

# Useful bounds (ignore NaNs)
def num_min(s): return int(np.nanmin(s))
def num_max(s): return int(np.nanmax(s))

AGE_MIN, AGE_MAX = num_min(df["Age"]), num_max(df["Age"])
FARE_MIN, FARE_MAX = float(np.nanmin(df["Fare"])), float(np.nanmax(df["Fare"]))
SIBSP_MIN, SIBSP_MAX = num_min(df["SibSp"]), num_max(df["SibSp"])
PARCH_MIN, PARCH_MAX = num_min(df["Parch"]), num_max(df["Parch"])

EMBARKED_OPTS = sorted(df["Embarked"].dropna().unique().tolist())
PCLASS_OPTS   = sorted(df["Pclass"].dropna().unique().tolist())
SEX_OPTS      = sorted(df["Sex"].dropna().unique().tolist())

# Defaults for resetting
DEFAULTS = dict(
    age=(max(0, AGE_MIN), AGE_MAX),  # some ages are 0/NaN in certain versions
    embarked=EMBARKED_OPTS,
    pclass=PCLASS_OPTS,
    sex=SEX_OPTS,
    fare=(max(0.0, FARE_MIN), min(100.0, FARE_MAX)),  # reasonable visible default
    sibsp=(SIBSP_MIN, SIBSP_MAX),
    parch=(PARCH_MIN, PARCH_MAX),
    survived="Both",
)

def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ---- UI ---------------------------------------------------------------------
st.title("Titanic Dataset Explorer")
st.caption("Interactively filter passengers and view distributions.")

with st.sidebar:
    st.header("Filter Options")

    age_filter = st.slider(
        "Age range",
        min_value=AGE_MIN, max_value=AGE_MAX, value=st.session_state.get("age", DEFAULTS["age"]),
        step=1, key="age",
        help="Passengers with missing Age are excluded by this range filter."
    )

    embarked_filter = st.multiselect(
        "Embarkation ports",
        options=EMBARKED_OPTS,
        default=st.session_state.get("embarked", DEFAULTS["embarked"]),
        key="embarked"
    )

    pclass_filter = st.multiselect(
        "Passenger classes",
        options=PCLASS_OPTS,
        default=st.session_state.get("pclass", DEFAULTS["pclass"]),
        key="pclass"
    )

    sex_filter = st.multiselect(
        "Sex",
        options=SEX_OPTS,
        default=st.session_state.get("sex", DEFAULTS["sex"]),
        key="sex"
    )

    fare_filter = st.slider(
        "Fare range",
        min_value=float(FARE_MIN), max_value=float(FARE_MAX),
        value=st.session_state.get("fare", DEFAULTS["fare"]),
        step=0.5, key="fare"
    )

    sibsp_filter = st.slider(
        "Siblings/Spouses aboard (SibSp)",
        min_value=SIBSP_MIN, max_value=SIBSP_MAX,
        value=st.session_state.get("sibsp", DEFAULTS["sibsp"]),
        step=1, key="sibsp"
    )

    parch_filter = st.slider(
        "Parents/Children aboard (Parch)",
        min_value=PARCH_MIN, max_value=PARCH_MAX,
        value=st.session_state.get("parch", DEFAULTS["parch"]),
        step=1, key="parch"
    )

    survived_filter = st.radio(
        "Survived",
        options=["Both", "Yes", "No"],
        index=["Both", "Yes", "No"].index(st.session_state.get("survived", DEFAULTS["survived"])),
        key="survived"
    )

    st.button("Clear filters", on_click=reset_filters, use_container_width=True)

# ---- Filtering logic (robust & clear) ---------------------------------------
mask = pd.Series(True, index=df.index)

# Age: exclude NaNs by requiring the value to be within range
mask &= df["Age"].between(age_filter[0], age_filter[1], inclusive="both")

# Categorical filters
if embarked_filter:
    mask &= df["Embarked"].isin(embarked_filter)
if pclass_filter:
    mask &= df["Pclass"].isin(pclass_filter)
if sex_filter:
    mask &= df["Sex"].isin(sex_filter)

# Numeric ranges
mask &= df["Fare"].between(fare_filter[0], fare_filter[1], inclusive="both")
mask &= df["SibSp"].between(sibsp_filter[0], sibsp_filter[1], inclusive="both")
mask &= df["Parch"].between(parch_filter[0], parch_filter[1], inclusive="both")

# Survived mapping
if survived_filter != "Both":
    target = 1 if survived_filter == "Yes" else 0
    mask &= df["Survived"].eq(target)

filtered_df = df.loc[mask].copy()

# ---- Summary & views --------------------------------------------------------
left, mid, right = st.columns(3)
left.metric("Passengers", f"{len(filtered_df):,}")
mid.metric("Survival Rate", f"{filtered_df['Survived'].mean()*100:.1f}%" if not filtered_df.empty else "—")
right.metric("Avg. Fare", f"${filtered_df['Fare'].mean():.2f}" if not filtered_df.empty else "—")

# st.subheader("Filtered Data")
# st.dataframe(filtered_df, use_container_width=True)

# Quick viz examples (uncomment as desired)
with st.expander("Quick charts", expanded = True):
    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(filtered_df.dropna(subset=["Age"]), x="Age", title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        fig_fare = px.histogram(filtered_df.dropna(subset=["Fare"]), x="Fare", title="Fare Distribution")
        st.plotly_chart(fig_fare, use_container_width=True)

    fig_surv = px.histogram(
        filtered_df, x="Pclass", color="Survived",
        barmode="group", histfunc="count",
        category_orders={"Survived": [0, 1]},
        title="Counts by Class & Survival"
    )
    fig_surv.update_layout(legend_title_text="Survived (0=No, 1=Yes)")
    st.plotly_chart(fig_surv, use_container_width=True)
