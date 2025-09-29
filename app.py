import pandas as pd
import streamlit as st

from apputil import survival_demographics, visualize_demographic, family_groups, last_names, visualize_families, visualize_family_size


# Load Titanic dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
)

# -------------------------------
# Titanic Visualization 1
# -------------------------------
st.header("Titanic Visualization 1")

table = survival_demographics(df)
question = (
    "Do women in first class have a higher survival rate "
    "than men in other classes?"
)

st.write(f"**Question:** {question}")
st.dataframe(table, use_container_width=True)

fig1 = visualize_demographic(table, question_text=question)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Titanic Visualization 2
# -------------------------------
st.header("Titanic Visualization 2")

# show family groups table
table2 = family_groups(df)
st.subheader("Family groups table")
st.dataframe(table2, use_container_width=True)

# show last_names
st.subheader("Last name counts (top 20)")
ln = last_names(df).head(20)
st.dataframe(ln.to_frame("count"), use_container_width=True)

# show question
question2 = "Do larger families pay higher average fares across passenger classes?"
st.write(f"**Question:** {question2}")

fig2 = visualize_families(table2, question_text=question2)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Titanic Visualization Bonus
# -------------------------------
st.header("Titanic Visualization Bonus — Age Division")

df_bonus = determine_age_division(df)
st.dataframe(df_bonus[["Pclass", "Age", "older_passenger"]].head(20), use_container_width=True)

fig3 = visualize_age_division(df_bonus)
st.plotly_chart(fig3, use_container_width=True)
