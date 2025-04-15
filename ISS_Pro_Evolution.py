# %% [markdown]
# # ISS Pro Evolution Player Database
#
# ![image](https://psxdatacenter.com/images/covers/U/I/SLUS-01014/SLUS-01014-F-ALL.jpg)

# %%
import pandas as pd
import scipy.stats as stats
from IPython.display import display

# %%
df = pd.read_parquet("original_data_compressed.parquet")

# %%
df = pd.read_parquet("original_data_compressed.parquet")

# %%
df.columns[10:25]

# %% [markdown]
# ## Categories

# %%
# fmt: off
categories = {
    'Attack': [
        'Attack', 'Strength', 'Stamina', 'Speed', 'Speed Up', 'Pass Acc',
        'Kick Pwr', 'Kick Acc', 'Jump Pwr', 'Head Acc', 'Technique',
        'Dribbling', 'Swerve', 'Aggression', "Normalized Height"
    ],
    'Defense': [
        'Defense', 'Strength', 'Stamina', 'Speed', 'Jump Pwr',
        'Head Acc', 'Aggression', "Normalized Height"
    ],
    'Power': [
        'Strength', 'Stamina', 'Kick Pwr', 'Jump Pwr', 'Aggression',
        "Normalized Height"
    ],
    'Speed': ['Speed', 'Speed Up', 'Dribbling'],
    'Technique':
    ['Pass Acc', 'Kick Acc', 'Head Acc', 'Technique', 'Dribbling', 'Swerve']
}

# %%
# fmt: off
per_pos = {
    'GK': [
        'Defense', 'Strength', 'Stamina', 'Speed', 'Jump Pwr', 'Aggression',
        'Normalized Height'
    ],
    'DF': [
        'Defense', 'Strength', 'Stamina', 'Speed', 'Jump Pwr', 'Aggression',
        'Speed', 'Jump Pwr', 'Head Acc', 'Normalized Height'
    ],
    'MF': [
        'Attack', 'Defense', 'Strength', 'Stamina', 'Speed', 'Speed Up',
        'Pass Acc', 'Kick Pwr', 'Kick Acc', 'Jump Pwr', 'Head Acc',
        'Technique', 'Dribbling', 'Swerve', 'Aggression',
    ],
    'FW': [
        'Attack', 'Strength', 'Stamina', 'Speed', 'Speed Up', 'Kick Pwr',
        'Kick Acc', 'Jump Pwr', 'Head Acc', 'Technique', 'Dribbling', 'Swerve',
        'Aggression', 'Normalized Height'
    ]
}


# %%
# fmt: on
def normalize(
    DATA_COL: pd.Series,
    range_start: float | int = 0,
    range_end: float | int = 1,
) -> pd.Series:
    r"""
    Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such that it
    is in the given range on the training set, e.g. between zero and one.

    The transformation is given by:

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean, unit
    variance scaling.

    Parameters
    ----------
    DATA_COL : pd.Series

    range_start : float | int, optional
                     , by default 0
    range_end : float | int, optional
                     , by default 1

    Returns
    -------
    pd.Series
        Normalized data
    """
    return range_start + (DATA_COL - DATA_COL.min()) * (range_end - range_start) / (
        DATA_COL.max() - DATA_COL.min()
    )


# %%
# fmt: off
def normalized_height(DATA: pd.DataFrame) -> pd.DataFrame:
    return DATA.assign(**{"Normalized Height": normalize(DATA["Height"], 1, 9)})


def raw_mean(DATA: pd.DataFrame) -> pd.DataFrame:
    return DATA.assign(
        **{
            "Raw Mean": DATA[
                [
                    "Attack", "Defense", "Strength", "Stamina", "Speed",
                    "Speed Up", "Pass Acc", "Kick Pwr", "Kick Acc", "Jump Pwr",
                    "Head Acc", "Technique", "Dribbling", "Swerve",
                    "Aggression", "Normalized Height",
                ]
            ].mean(axis=1)
        }
    )


# %%
# fmt: on
def ranking_stats(DATA: pd.DataFrame, stat="Mean") -> pd.DataFrame:
    # Fix
    if "Name" in DATA.columns:
        result = DATA.drop_duplicates(subset=["Name"])
    else:
        result = DATA

    return result.assign(
        **{
            f"{stat}_Z": stats.zscore(result[stat]),
            f"{stat}_Rank": result[stat].rank(ascending=False),
            f"{stat}_Percentile": result[stat].rank(pct=True),
        }
    ).sort_values(stat, ascending=False)


# %%
def categorize_per_pos(
    DATA: pd.DataFrame, pos: str, dict_cat: dict = per_pos
) -> pd.DataFrame:
    return (
        DATA.loc[DATA["POS"] == pos].assign(
            **{"Mean": DATA[dict_cat[pos]].mean(axis=1)}
        )
    ).pipe(
        lambda d: d.assign(
            **{
                "Cost/Mean": d["Cost"] / d["Mean"],
            }
        )
    )


# %%
def mean_per_pos(DATA: pd.DataFrame, dict_cat: dict = per_pos) -> pd.DataFrame:
    return DATA.assign(
        **dict(
            zip(
                [f"{p}_Mean" for p in per_pos.keys()],
                [DATA[stats].mean(axis=1) for stats in per_pos.values()],
            )
        )
    )


# %%
df.pipe(normalized_height).pipe(
    lambda d: d[
        [
            "Attack",
            "Defense",
            "Strength",
            "Stamina",
            "Speed",
            "Speed Up",
            "Pass Acc",
            "Kick Pwr",
            "Kick Acc",
            "Jump Pwr",
            "Head Acc",
            "Technique",
            "Dribbling",
            "Swerve",
            "Aggression",
            "Normalized Height",
        ]
    ]
    > 6
).sum(axis=1).sort_values(ascending=False)

# %%
df.pipe(normalized_height).assign(
    **{
        "Top Count": lambda d: (
            d[
                [
                    "Attack",
                    "Defense",
                    "Strength",
                    "Stamina",
                    "Speed",
                    "Speed Up",
                    "Pass Acc",
                    "Kick Pwr",
                    "Kick Acc",
                    "Jump Pwr",
                    "Head Acc",
                    "Technique",
                    "Dribbling",
                    "Swerve",
                    "Aggression",
                    "Normalized Height",
                ]
            ]
            > 6
        ).sum(axis=1)
    }
).sort_values(by="Top Count", ascending=False)

# %%
for p in ["GK", "DF", "MF", "FW"]:
    print(p)
    display(
        df.pipe(normalized_height)
        .pipe(lambda d: d.assign(**{"Region": d["Region"].astype(str)}))
        .pipe(mean_per_pos)
        # .pipe(ranking_stats, "MF_Mean")
        .loc[lambda d: d["Region"] == "Africa"]
        .assign(
            **{
                "Overall": lambda d: d[
                    ["GK_Mean", "DF_Mean", "MF_Mean", "FW_Mean"]
                ].sum(axis=1)
            }
        )
        .loc[lambda e: e["Country"] == "Cameroon"]
        .sort_values(f"{p}_Mean", ascending=False)
        # .loc[lambda f: f.groupby(["POS", "GK_Mean", "DF_Mean", "MF_Mean", "FW_Mean"], observed=True)['Overall'].idxmax()]
        # .groupby("Team", observed=False)
        # .agg({"Overall": "mean"})
        # .sort_values("Overall", ascending=False)
        # .dropna()
    )

# %%
pd.concat(
    [df.pipe(normalized_height).pipe(categorize_per_pos, p) for p in per_pos.keys()][
        2:3
    ]
).pipe(raw_mean).loc[lambda d: d["Cost"].notna()].pipe(ranking_stats).head(11)

# %%
for p in per_pos.keys():
    print(p)
    display(
        df.loc[df["Country"] == "Cameroon"]
        .pipe(normalized_height)
        .pipe(categorize_per_pos, p)
        .pipe(raw_mean)
        .loc[lambda d: d["Cost"].notna()]
        .pipe(ranking_stats)
        .pipe(pd_col_to_front, "Mean")
    )

# %%
pd.concat(
    [df.pipe(normalized_height).pipe(categorize_per_pos, p) for p in per_pos.keys()]
).pipe(raw_mean).loc[lambda _: _["Starter/Sub"] == "Starter"].pipe(
    lambda d: pd.crosstab(
        index=d["Team"],
        columns=d["POS"],
        values=d["Mean"],
        aggfunc="mean",
    )
).assign(
    **{"OVERALL": lambda d_: d_.mean(axis=1)}
).sort_values(
    "OVERALL", ascending=False
).pipe(
    ranking_stats, "OVERALL"
)

# %%
data_with_means = pd.concat(
    [df.pipe(normalized_height).pipe(categorize_per_pos, p) for p in per_pos.keys()]
).pipe(raw_mean)

# %%
{k: i for i, k in enumerate(["GK", "DF", "MF", "FW"])}

# %%
data_with_means.loc[lambda d: d["Mean"] > d["Mean"].quantile(0.95)].sort_values(
    "Mean", ascending=False
).drop_duplicates(subset=["Name"]).assign(
    **{"POS": lambda d_: d_["POS"].astype(str)}
).sort_values(
    by="POS",
    key=lambda x: x.map({k: i for i, k in enumerate(["GK", "DF", "MF", "FW"])}),
).query(
    # "POS == 'DF'"
    "Region == 'Africa'"
)

# %%
df.iloc[:, 10:25].loc[lambda d: ((d > 9) | (d < 1)).any(axis=1)]

# %%
for a in sorted(df["Age"].unique()):
    print(a)
    display(
        pd.concat(
            [
                df.pipe(normalized_height).pipe(categorize_per_pos, p)
                for p in per_pos.keys()
            ]
        )
        .pipe(raw_mean)
        .loc[lambda d: d["Age"] == a]
        .pipe(ranking_stats)
        .head(1)
    )

# %%
for col in [f"{k}_Mean" for k in categories.keys()]:
    print(col)
    display(
        df.loc[(df["Country/Club"] == "Country")]
        .pipe(
            lambda d: d.assign(
                **dict(
                    zip(
                        [f"{k}_Mean" for k in categories.keys()],
                        [
                            d.pipe(normalized_height)[stat].mean(axis=1)
                            for stat in categories.values()
                        ],
                    )
                )
            ).assign(**{"Team": lambda e: e["Team"].astype(str)})
        )
        .loc[lambda e: e[col] > e[col].quantile(0.99)]
        .sort_values(col, ascending=False)
    )


# %%
def apply_func_to_columns(df, func):
    result = df.apply(func)
    if func.__name__ == "max":
        index = df.idxmax()
    elif func.__name__ == "min":
        index = df.idxmin()
    else:
        median = result.median()
        index = result[result == median].index.tolist()
    return pd.concat(
        [result, pd.Series(index, index=result.index)],
        axis=1,
        keys=["result", "index"],
    )
