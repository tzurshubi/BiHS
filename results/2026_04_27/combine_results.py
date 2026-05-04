import pandas as pd
from pathlib import Path

base = Path(__file__).parent

BLOCKED_PCTS = ["20%", "16%", "12%", "8%"]
METRICS = ["Expansions", "Time [ms]"]
ALG_ORDER = ["XDFBnB", "BiXDFBnB", "XIDA", "BiXIDA"]

# Load all CSVs
dfs = []
for alg_family in ["DFBnB", "IDA"]:
    for domain in ["LSP_Grids", "Snake_Grids"]:
        for la in [1, 2, 3, 4]:
            path = base / alg_family / domain / f"{la}la" / "output.csv"
            if not path.exists():
                print(f"Missing: {path}")
                continue
            df = pd.read_csv(path)
            df["Domain"] = domain
            df["Lookahead"] = la
            dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)

# Keep only the 4 relevant blocked percentages
metric_cols = [f"{p} {m}" for p in BLOCKED_PCTS for m in METRICS]
raw = raw[["Grid", "Algorithm", "Domain", "Lookahead"] + metric_cols]

# Replace "-" with NaN; drop rows where all metrics are empty
raw[metric_cols] = raw[metric_cols].replace("-", pd.NA)
raw = raw.dropna(subset=metric_cols, how="all")

# Convert to numeric
raw[metric_cols] = raw[metric_cols].apply(pd.to_numeric, errors="coerce")

# Melt to long format then pivot to MultiIndex
melted = raw.melt(
    id_vars=["Grid", "Algorithm", "Domain", "Lookahead"],
    value_vars=metric_cols,
    var_name="col",
    value_name="value",
)
melted[["Blocked%", "Metric"]] = melted["col"].str.extract(r"(\d+%)\s+(.*)")
melted = melted.drop(columns="col")

for domain in ["LSP_Grids", "Snake_Grids"]:
    d = melted[melted["Domain"] == domain]

    pivot = d.pivot_table(
        index=["Grid", "Algorithm", "Lookahead"],
        columns=["Blocked%", "Metric"],
        values="value",
        aggfunc="first",
    )

    # Reorder columns: 20% → 16% → 12% → 8%, each with Expansions then Time [ms]
    col_order = pd.MultiIndex.from_tuples([(p, m) for p in BLOCKED_PCTS for m in METRICS])
    pivot = pivot.reindex(columns=col_order)

    # Reorder rows: Grid (ascending by dimensions), Algorithm (custom), Lookahead (ascending)
    pivot = pivot.reset_index()
    pivot["_g0"] = pivot["Grid"].str.extract(r"^(\d+)").astype(int)
    pivot["_g1"] = pivot["Grid"].str.extract(r"x(\d+)$").astype(int)
    pivot["Algorithm"] = pd.Categorical(pivot["Algorithm"], categories=ALG_ORDER, ordered=True)
    pivot = (
        pivot.sort_values(["_g0", "_g1", "Algorithm", "Lookahead"])
        .drop(columns=["_g0", "_g1"])
        .set_index(["Grid", "Algorithm", "Lookahead"])
    )

    # Convert to integer where possible (no decimals in expansions/time)
    pivot = pivot.astype("Int64")

    out_path = base / f"table_{domain}.csv"
    pivot.to_csv(out_path)
    print(f"Saved {out_path}  ({len(pivot)} rows x {len(pivot.columns)} cols)")
