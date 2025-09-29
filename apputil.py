import plotly.express as px
import pandas as pd

# update/add code below ...

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    return Pclass, Sex, age_group:
    - n_passengers: total number of passengers
    - n_survivors: number of survivors
    - survival_rate: survival rate
    """
    # create age group（Child/Teen/Adult/Senior）
    bins   = [0, 12, 19, 59, 200]
    labels = ["Child", "Teen", "Adult", "Senior"]
    age_group = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    tmp = df.assign(age_group=age_group)

    # group by class, sez, age_group
    out = (
        tmp.groupby(["Pclass", "Sex", "age_group"])
           .agg(
               n_passengers=("PassengerId", "count"),
               n_survivors=("Survived", "sum"),
           )
           .reset_index()
    )
    out["survival_rate"] = out["n_survivors"] / out["n_passengers"]

    # sort
    order_map = {"Child": 0, "Teen": 1, "Adult": 2, "Senior": 3}
    out["age_order"] = out["age_group"].map(order_map)
    out = out.sort_values(["Pclass", "Sex", "age_order"]).drop(columns="age_order")

    return out



def visualize_demographic(table: pd.DataFrame, question_text: str | None = None):
    """
    Visualize survival rate by age group, sex, class
    """
    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        category_orders={"age_group": ["Child", "Teen", "Adult", "Senior"]},
        labels={"age_group":"Age group", "survival_rate":"Survival rate", "Pclass":"Class"},
        title=question_text or "Survival Rate by Age Group, Sex, and Class"
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    create family_size = SibSp + Parch + 1,
    family_size * Pclass:
      - n_passengers
      - avg_fare
      - min_fare / max_fare
    sort with class、family_size
    """
    tmp = df.copy()
    tmp["family_size"] = tmp["SibSp"] + tmp["Parch"] + 1

    out = (
        tmp.groupby(["family_size", "Pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
        .sort_values(["Pclass", "family_size"])
    )
    return out

def last_names(df: pd.DataFrame) -> pd.Series:
    """
    get family name then return value_counts Series
    index=last_name, value=count
    """
    last = df["Name"].str.split(",", n=1).str[0].str.strip()
    return last.value_counts()

def visualize_families(df=None):
    if df is None:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
        )
    
    # family size
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    
    # groupby family_size and class
    grouped = df.groupby(["family_size", "Pclass"]).agg(
        n_passengers=("PassengerId", "count"),
        avg_fare=("Fare", "mean"),
        min_fare=("Fare", "min"),
        max_fare=("Fare", "max"),
    ).reset_index()
    
    # plot
    fig = px.bar(
        grouped,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        barmode="group",
        title="Average Fare by Family Size and Passenger Class"
    )
    return fig