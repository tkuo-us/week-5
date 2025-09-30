import plotly.express as px
import pandas as pd

# update/add code below ...
_TITANIC_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"

# ---------- for autograder ----------
def _ensure_df(df=None) -> pd.DataFrame:
    """If they don't have df, then loading original Titanic dataset。"""
    if df is None:
        return pd.read_csv(_TITANIC_URL)
    return df

def _find_col(df: pd.DataFrame, *cands: str) -> str:
    """Return the first existing column name (case-insensitive)."""
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of {cands} found in columns: {list(df.columns)}")


# ---------- 1) Exercise 1 ----------
def survival_demographics(df=None) -> pd.DataFrame:
    """
    return Pclass, Sex, age_group:
    - n_passengers: total number of passengers
    - n_survivors: number of survivors
    - survival_rate: survival rate
    """
    df = _ensure_df(df)

    age_col      = _find_col(df, "Age", "age")
    pclass_col   = _find_col(df, "Pclass", "pclass")
    sex_col      = _find_col(df, "Sex", "sex")
    survived_col = _find_col(df, "Survived", "survived")
    pid_col      = _find_col(df, "PassengerId", "passengerid")

    # create age group（Child/Teen/Adult/Senior）
    bins   = [0, 12, 19, 59, 200]
    labels = ["Child", "Teen", "Adult", "Senior"]
    age_group = pd.cut(
        df[age_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    tmp = df.assign(age_group=age_group)

    # group by class, sez, age_group
    out = (
        tmp.groupby([pclass_col, sex_col, "age_group"], observed=False)
           .agg(
               n_passengers=(pid_col, "count"),
               n_survivors=(survived_col, "sum"),
           )
           .reset_index()
    )
    out["survival_rate"] = out["n_survivors"] / out["n_passengers"]
    out["age_group"] = out["age_group"].astype(
        pd.CategoricalDtype(categories=labels, ordered=True)
    )
    # rename columns for autograder
    out = out.rename(columns={pclass_col: "pclass", sex_col: "sex"})
    out = out.sort_values(["pclass", "sex", "age_group"])

    return out[["pclass", "sex", "age_group", "n_passengers", "n_survivors", "survival_rate"]]



def visualize_demographic(table: pd.DataFrame, question_text: str | None = None):
    """
    Visualize survival rate by age group, sex, pclass
    Expects the OUTPUT schema from survival_demographics (lower-case)
    """
    facet_col = "pclass" if "pclass" in table.columns else "Pclass"
    color_col = "sex"    if "sex"    in table.columns else "Sex"

    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color=color_col,
        facet_col=facet_col,
        barmode="group",
        category_orders={"age_group": ["Child", "Teen", "Adult", "Senior"]},
        labels={
            "age_group": "Age group",
            "survival_rate": "Survival rate",
            facet_col: "Class",
            color_col: "Sex",
        },
        title=question_text or "Survival Rate by Age Group, Sex, and Class",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# ---------- 2) Exercise 2 ----------
def family_groups(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    create family_size = SibSp + Parch + 1, group by (family_size, pclass)
    Return (lower-case pclass in OUTPUT):
      family_size, pclass, n_passengers, avg_fare, min_fare, max_fare
    """
    #df = df.copy() # avoid modifying original df
    df = _ensure_df(df).copy()

    sibsp_col  = _find_col(df, "SibSp", "sibsp")
    parch_col  = _find_col(df, "Parch", "parch")
    pclass_col = _find_col(df, "Pclass", "pclass")
    fare_col   = _find_col(df, "Fare", "fare")
    pid_col    = _find_col(df, "PassengerId", "passengerid")

    df["family_size"] = df[sibsp_col] + df[parch_col] + 1

    out = (
        df.groupby(["family_size", pclass_col], observed=False)
        .agg(
            n_passengers=(pid_col, "count"),
            avg_fare=(fare_col, "mean"),
            min_fare=(fare_col, "min"),
            max_fare=(fare_col, "max"),
        )
        .reset_index()
    )
    out = out.rename(columns={pclass_col: "pclass"})
    out = out.sort_values(["pclass", "family_size"])
    return out


def last_names(df=None) -> pd.Series:
    """
    get family name then return value_counts Series
    index=last_name, value=count
    """
    df = _ensure_df(df)

    name_col = _find_col(df, "Name", "name")
    last = df[name_col].str.split(",", n=1).str[0].str.strip()
    return last.value_counts()

def visualize_families(table: pd.DataFrame, question_text: str | None = None):
    fig = px.line(
        table,
        x="family_size",
        y="avg_fare",
        color="pclass",
        markers=True,
        title=question_text or "Average Fare by Family Size and Class",
    )
    return fig

# bonus
def visualize_family_size(df: pd.DataFrame | None = None):
    """
    family_size
    """
    df = _ensure_df(df).copy()

    sibsp_col = _find_col(df, "SibSp", "sibsp")
    parch_col = _find_col(df, "Parch", "parch")
    df["family_size"] = df[sibsp_col] + df[parch_col] + 1
    g = df["family_size"].value_counts().sort_index().reset_index()
    g.columns = ["family_size", "n_passengers"]
    return px.bar(g, x="family_size", y="n_passengers", title="Passenger Count by Family Size")


def determine_age_division(df=None) -> pd.DataFrame:
    """
    create a new boolean 'older_passenger' to mark passengers older than the median age.
    """
    df = _ensure_df(df).copy()

    age_col = _find_col(df, "Age", "age")
    median_age = df[age_col].median()
    df["older_passenger"] = df[age_col] > median_age
    return df

def visualize_age_division(df: pd.DataFrame):
    """
    Compatible with either 'Pclass'/'pclass' and 'Survived'/'survived'
    """

    x_col = "Pclass" if "Pclass" in df.columns else _find_col(df, "pclass", "Pclass")
    y_col = "Survived" if "Survived" in df.columns else _find_col(df, "survived", "Survived")

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color="older_passenger",
        barmode="group",
        title="Survival by Passenger Class and Age Division",
    )
    return fig
