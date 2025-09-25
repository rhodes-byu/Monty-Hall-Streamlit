# app.py
import os
import io
import uuid
import time
import sqlite3
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

###############################################################################
# Persistence (SQLite) for multi-user / multi-session aggregate statistics
###############################################################################

DB_PATH = "monty.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            player_id TEXT NOT NULL,
            first_pick INTEGER NOT NULL CHECK(first_pick IN (1,2,3)),
            car_door INTEGER NOT NULL CHECK(car_door IN (1,2,3)),
            monty_open INTEGER NOT NULL CHECK(monty_open IN (1,2,3)),
            switched INTEGER NOT NULL CHECK(switched IN (0,1)),
            final_pick INTEGER NOT NULL CHECK(final_pick IN (1,2,3)),
            won INTEGER NOT NULL CHECK(won IN (0,1))
        )
    """)
    conn.commit()
    conn.close()

def record_play(row):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO plays (ts, player_id, first_pick, car_door, monty_open, switched, final_pick, won)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (row["ts"], row["player_id"], row["first_pick"], row["car_door"],
          row["monty_open"], row["switched"], row["final_pick"], row["won"]))
    conn.commit()
    conn.close()

def fetch_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM plays ORDER BY id ASC", conn)
    conn.close()
    return df

###############################################################################
# Utility: Wilson confidence interval for a binomial proportion (nice for small n)
###############################################################################
def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half_width = (z*np.sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    return phat, max(0.0, center - half_width), min(1.0, center + half_width)

###############################################################################
# Game mechanics
###############################################################################
def place_car():
    return int(np.random.choice([1, 2, 3]))

def monty_opens(chosen, car):
    # Monty opens a goat door that is not the player's chosen door
    doors = [1, 2, 3]
    goats = [d for d in doors if d != car]
    options = [d for d in goats if d != chosen]
    return int(np.random.choice(options))

def other_closed(chosen, monty_open):
    return [d for d in [1,2,3] if d not in (chosen, monty_open)][0]

###############################################################################
# App state init
###############################################################################
st.set_page_config(page_title="Monty Hall Simulator", page_icon="🚪", layout="wide")

# CSS: color primary (green) vs secondary (light gray) buttons
st.markdown("""
<style>
/* Green primary button */
div.stButton > button[kind="primary"] {
    background-color: #22c55e !important; /* green */
    color: #ffffff !important;
}
/* Light gray secondary button */
div.stButton > button[kind="secondary"] {
    background-color: #e5e7eb !important; /* light gray */
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
}
</style>
""", unsafe_allow_html=True)

init_db()

if "player_id" not in st.session_state:
    # Create a stable per-session identifier (not a cookie, just to separate personal stats if desired)
    st.session_state.player_id = str(uuid.uuid4())

if "round_state" not in st.session_state:
    st.session_state.round_state = "setup"  # setup -> picked -> revealed -> finished
if "car_door" not in st.session_state:
    st.session_state.car_door = None
if "first_pick" not in st.session_state:
    st.session_state.first_pick = None
if "monty_open" not in st.session_state:
    st.session_state.monty_open = None
if "final_pick" not in st.session_state:
    st.session_state.final_pick = None
if "switched" not in st.session_state:
    st.session_state.switched = 0

if "car_image" not in st.session_state:
    st.session_state.car_image = None
if "goat_image" not in st.session_state:
    st.session_state.goat_image = None

###############################################################################
# Sidebar: Controls & Images
###############################################################################
with st.sidebar:
    st.header("Controls")
    strategy = st.radio(
        "Switch strategy this round:",
        ["Decide after reveal", "Always SWITCH", "Always STAY"],
        index=0
    )
    st.caption("Choose your strategy for the current round. You can change it any time.")

    st.markdown("---")
    st.subheader("Images (optional)")
    car_file = st.file_uploader("Car image", type=["png", "jpg", "jpeg", "gif"], key="car_uploader")
    goat_file = st.file_uploader("Goat image", type=["png", "jpg", "jpeg", "gif"], key="goat_uploader")

    if car_file is not None:
        st.session_state.car_image = car_file.read()
    if goat_file is not None:
        st.session_state.goat_image = goat_file.read()

    st.markdown("If no images are uploaded, emoji placeholders 🚗 / 🐐 will be used.")

###############################################################################
# Title
###############################################################################
st.title("🚪 Monty Hall Simulator")
st.write("Play the game, choose whether to switch or stay, and see empirical win rates split by strategy. Multiple people can play—results are stored and aggregated.")

###############################################################################
# Layout
###############################################################################
left, right = st.columns([1.1, 1])

###############################################################################
# Left: Game interface
###############################################################################
with left:
    st.subheader("Game")
    st.caption("1) Pick a door → 2) Monty reveals a goat → 3) Decide whether to switch → 4) Reveal outcome")

    # Determine button type (green only when game is finished and needs reset, gray otherwise)
    needs_new_round = st.session_state.round_state == "finished"

    btn_type = "primary" if needs_new_round else "secondary"

    # Start/Reset button with dynamic styling
    if st.button("Start / Reset Round", use_container_width=True, type=btn_type):
        st.session_state.round_state = "setup"
        st.session_state.car_door = place_car()
        st.session_state.first_pick = None
        st.session_state.monty_open = None
        st.session_state.final_pick = None
        st.session_state.switched = 0

    # Initialize car location if needed
    if st.session_state.round_state == "setup" and st.session_state.car_door is None:
        st.session_state.car_door = place_car()

    # Door selection UI
    st.markdown("#### Choose your door")
    choice_cols = st.columns(3)
    for idx, door in enumerate([1, 2, 3]):
        with choice_cols[idx]:
            disabled = st.session_state.first_pick is not None
            if st.button(f"Pick Door {door}", disabled=disabled, key=f"pick_{door}", use_container_width=True):
                if st.session_state.round_state in ("setup", "finished"):
                    # starting a fresh round if user clicks after finish
                    if st.session_state.round_state == "finished":
                        st.session_state.car_door = place_car()
                    st.session_state.round_state = "picked"
                st.session_state.first_pick = door
                # Monty opens immediately after first pick
                st.session_state.monty_open = monty_opens(st.session_state.first_pick, st.session_state.car_door)

    # Show reveal (Monty opens a goat door)
    if st.session_state.first_pick is not None and st.session_state.monty_open is not None:
        st.markdown("#### Monty reveals a goat behind:")
        st.info(f"Door {st.session_state.monty_open}")

        # For "Always SWITCH/STAY", commit the final choice now
        if st.session_state.round_state == "picked":
            if strategy == "Always SWITCH":
                st.session_state.final_pick = other_closed(st.session_state.first_pick, st.session_state.monty_open)
                st.session_state.switched = 1
                st.session_state.round_state = "finished"
            elif strategy == "Always STAY":
                st.session_state.final_pick = st.session_state.first_pick
                st.session_state.switched = 0
                st.session_state.round_state = "finished"
            else:
                # Decide after reveal -> let them choose now
                st.session_state.round_state = "revealed"

    # If "Decide after reveal", present the choice
    if st.session_state.round_state == "revealed":
        st.markdown("#### Stay or Switch?")
        with st.form("final_decision"):
            current_choice = st.radio(
                "Your decision for THIS round:",
                ["Stay with original", "Switch to the other unopened door"],
                horizontal=True,
                key="decision_radio"
            )
            submitted = st.form_submit_button("Lock in final choice", type="primary")

        if submitted:
            if current_choice == "Stay with original":
                st.session_state.final_pick = st.session_state.first_pick
                st.session_state.switched = 0
            else:
                st.session_state.final_pick = other_closed(
                    st.session_state.first_pick, st.session_state.monty_open
                )
                st.session_state.switched = 1
            st.session_state.round_state = "finished"

    # Outcome reveal + visual
    if st.session_state.round_state == "finished" and st.session_state.final_pick is not None:
        won = int(st.session_state.final_pick == st.session_state.car_door)

        # Record to DB
        record_play({
            "ts": datetime.utcnow().isoformat(),
            "player_id": st.session_state.player_id,
            "first_pick": st.session_state.first_pick,
            "car_door": st.session_state.car_door,
            "monty_open": st.session_state.monty_open,
            "switched": st.session_state.switched,
            "final_pick": st.session_state.final_pick,
            "won": won
        })

        st.success(f"Result: **{'WIN 🎉' if won else 'Loss 😿'}**")
        st.write(f"Car was behind **Door {st.session_state.car_door}**.")
        st.write(f"You ended on **Door {st.session_state.final_pick}** ({'switched' if st.session_state.switched else 'stayed'}).")

        # Visual depiction for each door
        def door_label(d):
            if st.session_state.car_door == d:
                return "CAR"
            else:
                return "GOAT"

        def show_image(label):
            if label == "CAR":
                if st.session_state.car_image:
                    st.image(io.BytesIO(st.session_state.car_image), use_container_width=True)
                else:
                    st.markdown("<div style='font-size:64px; text-align:center;'>🚗</div>", unsafe_allow_html=True)
            else:
                if st.session_state.goat_image:
                    st.image(io.BytesIO(st.session_state.goat_image), use_container_width=True)
                else:
                    st.markdown("<div style='font-size:64px; text-align:center;'>🐐</div>", unsafe_allow_html=True)

        st.markdown("#### Door Reveal")
        dcols = st.columns(3)
        for i, d in enumerate([1,2,3]):
            with dcols[i]:
                st.markdown(f"**Door {d}**")
                if d == st.session_state.monty_open or st.session_state.round_state == "finished":
                    show_image(door_label(d))
                else:
                    st.caption("Closed")

        st.caption("Start/Reset Round in the left panel to play again.")

###############################################################################
# Right: Analytics & Empirical Distributions
###############################################################################
with right:
    st.subheader("Aggregate Results")
    df = fetch_df()

    if df.empty:
        st.info("No games recorded yet. Play a round to populate statistics.")
    else:
        # Basic counts
        total_plays = len(df)
        switches = int(df["switched"].sum())
        stays = total_plays - switches
        wins = int(df["won"].sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total games", f"{total_plays:,}")
        with c2:
            st.metric("Times switched", f"{switches:,}")
        with c3:
            st.metric("Total wins", f"{wins:,}")

        # Conditional distributions: wins given switch vs stay
        grouped = df.groupby("switched").agg(
            n=("won", "size"),
            wins=("won", "sum")
        ).reset_index()

        # Ensure both rows exist
        if 0 not in grouped["switched"].values:
            grouped = pd.concat([grouped, pd.DataFrame([{"switched":0,"n":0,"wins":0}])], ignore_index=True)
        if 1 not in grouped["switched"].values:
            grouped = pd.concat([grouped, pd.DataFrame([{"switched":1,"n":0,"wins":0}])], ignore_index=True)
        grouped = grouped.sort_values("switched")

        # Compute rates + Wilson intervals
        rows = []
        for _, r in grouped.iterrows():
            k, n = int(r["wins"]), int(r["n"])
            p, low, high = wilson_interval(k, n)
            rows.append({
                "Strategy": "Switch" if int(r["switched"])==1 else "Stay",
                "Plays": n,
                "Wins": k,
                "Win Rate": p if not np.isnan(p) else 0.0,
                "CI Low": low if not np.isnan(low) else 0.0,
                "CI High": high if not np.isnan(high) else 0.0
            })
        rates = pd.DataFrame(rows)

        # st.markdown("#### Empirical win rate by strategy (with 95% CI)")
        # base = alt.Chart(rates).encode(x=alt.X("Strategy:N", title=None))
        # bars = base.mark_bar().encode(y=alt.Y("Win Rate:Q", scale=alt.Scale(domain=[0,1])))
        # error = base.mark_rule().encode(
        #     y="CI Low:Q",
        #     y2="CI High:Q",
        #     tooltip=["Strategy","Win Rate","CI Low","CI High","Plays","Wins"]
        # )
        # st.altair_chart(bars + error, use_container_width=True)

        # st.caption("Bars show empirical win rate. Vertical lines show 95% Wilson confidence intervals.")

        # Stacked wins/losses per strategy
        # st.markdown("#### Wins vs Losses by strategy")
        # stacked = pd.DataFrame({
        #     "Strategy": ["Stay", "Stay", "Switch", "Switch"],
        #     "Outcome": ["Win", "Loss", "Win", "Loss"],
        #     "Count": [
        #         int(df.loc[df["switched"]==0,"won"].sum()),
        #         int((df["switched"]==0).sum() - int(df.loc[df["switched"]==0,"won"].sum())),
        #         int(df.loc[df["switched"]==1,"won"].sum()),
        #         int((df["switched"]==1).sum() - int(df.loc[df["switched"]==1,"won"].sum()))
        #     ]
        # })
        # stacked_chart = alt.Chart(stacked).mark_bar().encode(
        #     x="Strategy:N",
        #     y="Count:Q",
        #     color="Outcome:N",
        #     order=alt.Order("Outcome", sort="ascending"),
        #     tooltip=["Strategy","Outcome","Count"]
        # )
        # st.altair_chart(stacked_chart, use_container_width=True)

        # Running win rate over time per strategy (optional but nice)
        st.markdown("#### Running win rate over time (by strategy)")
        df["ts_dt"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts_dt").reset_index(drop=True)
        df["idx"] = df.index + 1
        # Separate running means
        def running_rate(x):
            return x.expanding().mean()

        df["win_rate_overall"] = running_rate(df["won"])
        for s in (0,1):
            mask = df["switched"]==s
            df.loc[mask, f"win_rate_{s}"] = running_rate(df.loc[mask, "won"])
        melt_cols = ["win_rate_overall","win_rate_0","win_rate_1"]
        plot_df = df[["idx"] + melt_cols].melt("idx", value_name="rate", var_name="series").dropna()
        mapper = {
            "win_rate_overall":"Overall",
            "win_rate_0":"Stay",
            "win_rate_1":"Switch"
        }
        plot_df["series"] = plot_df["series"].map(mapper)

        line = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("idx:Q", title="Game count"),
            y=alt.Y("rate:Q", title="Running win rate", scale=alt.Scale(domain=[0,1])),
            color=alt.Color("series:N", title="Series"),
            tooltip=["series","idx","rate"]
        )
        st.altair_chart(line, use_container_width=True)

        # Raw data & download
        with st.expander("Show raw plays data"):
            st.dataframe(df.drop(columns=["car_door","monty_open"]), use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name="monty_plays.csv", mime="text/csv")

###############################################################################
# Footer / Help
###############################################################################
st.markdown("---")
st.markdown(
"""
**How it works (conditional probability view):** Initially your door has a 1/3 chance of hiding the car.
Monty—who knows where the car is—*must* open a different door with a goat. That action concentrates the remaining
2/3 probability mass onto the other unopened door, so switching wins with probability ~2/3 in the long run.
"""
)
