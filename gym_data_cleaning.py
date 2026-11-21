"""
Gym membership data cleaning
Uses only numpy and pandas.

Usage:
    py -3.12 gym_data_cleaning.py
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INPUT_PATH = "gym_membership_messy_duplicate_ids.csv"
OUTPUT_PATH = "gym_membership_cleaned.csv"

# Fix the confusing / inconsistent column names
RENAME_COLUMNS = {"age": "age", "abonoment_type": "membership_type"}

WEEK_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
VALID_GENDERS = {"Female", "Male"}
VALID_MEMBERSHIPS = {"Standard", "Premium"}

# Text used where a value is structurally absent (e.g. no group lesson)
NO_VALUE_LABEL = "None"

log = []  # running list of (step, rows affected) for the final report


def note(step, count):
    log.append((step, int(count)))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_bool(series):
    """Convert True/False-like values (bool, 'True', 'yes', 1 ...) to nullable boolean."""
    mapping = {
        "true": True, "t": True, "yes": True, "y": True, "1": True,
        "false": False, "f": False, "no": False, "n": False, "0": False,
    }
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
        .astype("boolean")
    )


def normalise_list(value, sort_key=None):
    """'b,  a ,a' -> 'a, b'. Missing/empty -> NaN."""
    if pd.isna(value):
        return np.nan
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    items = list(dict.fromkeys(items))  # drop repeats, keep order
    if not items:
        return np.nan
    items = sorted(items, key=sort_key) if sort_key else sorted(items)
    return ", ".join(items)


def calc_age(birthday, ref_date):
    """Completed years between birthday and ref_date (vectorised)."""
    before_bday = (birthday.dt.month * 100 + birthday.dt.day) > (
        ref_date.month * 100 + ref_date.day
    )
    return ref_date.year - birthday.dt.year - before_bday.astype(int)


def infer_reference_date(birthday, age, start="2024-01-01", end="2025-12-31"):
    """Find the date for which the stored Age best matches the birthdays."""
    candidates = pd.date_range(start, end)
    b_md = (birthday.dt.month * 100 + birthday.dt.day).to_numpy()[:, None]
    c_md = (candidates.month * 100 + candidates.day).to_numpy()[None, :]
    ages = (
        candidates.year.to_numpy()[None, :]
        - birthday.dt.year.to_numpy()[:, None]
        - (b_md > c_md)
    )
    mismatches = (ages != age.to_numpy()[:, None]).sum(axis=0)
    return candidates[int(np.argmin(mismatches))], int(mismatches.min())


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
n_start = len(df)

# --------------------------------------------------------------------------
# 2. Column names and basic text hygiene
# --------------------------------------------------------------------------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.rename(columns=RENAME_COLUMNS)

for col in df.columns:
    df[col] = df[col].str.strip()

df = df.replace(
    {
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "NULL": np.nan,
        "null": np.nan,
        "N/A": np.nan,
        "n/a": np.nan,
    }
)

# --------------------------------------------------------------------------
# 3. Remove exact duplicate rows and empty rows
# --------------------------------------------------------------------------
before = len(df)
df = df.dropna(how="all")
note("Fully empty rows dropped", before - len(df))

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
note("Exact duplicate rows dropped", before - len(df))

# --------------------------------------------------------------------------
# 4. Data types
# --------------------------------------------------------------------------
df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
df["visit_per_week"] = pd.to_numeric(df["visit_per_week"], errors="coerce").astype("Int64")
df["avg_time_in_gym"] = pd.to_numeric(df["avg_time_in_gym"], errors="coerce").astype("Int64")

for col in ["attend_group_lesson", "drink_abo", "personal_training", "uses_sauna"]:
    df[col] = to_bool(df[col])

note("Unparseable birthdays", df["birthday"].isna().sum())

# --------------------------------------------------------------------------
# 5. Categorical columns
# --------------------------------------------------------------------------
df["gender"] = df["gender"].str.title()
df["membership_type"] = df["membership_type"].str.title()

note("Invalid gender values set to missing", (~df["gender"].isin(VALID_GENDERS)).sum())
note(
    "Invalid membership types set to missing",
    (~df["membership_type"].isin(VALID_MEMBERSHIPS)).sum(),
)

df.loc[~df["gender"].isin(VALID_GENDERS), "gender"] = np.nan
df.loc[~df["membership_type"].isin(VALID_MEMBERSHIPS), "membership_type"] = np.nan

df["name_personal_trainer"] = df["name_personal_trainer"].str.title()

# --------------------------------------------------------------------------
# 6. Multi-value text columns (days, lessons, drinks)
# --------------------------------------------------------------------------
df["days_per_week"] = df["days_per_week"].str.title().apply(
    lambda v: normalise_list(
        v,
        sort_key=lambda d: WEEK_ORDER.index(d[:3]) if d[:3] in WEEK_ORDER else 99,
    )
)
df["fav_group_lesson"] = df["fav_group_lesson"].apply(normalise_list)
df["fav_drink"] = df["fav_drink"].str.lower().apply(normalise_list)

# --------------------------------------------------------------------------
# 7. Duplicate IDs
#    Duplicated IDs belong to *different* members (no repeated rows) and the
#    same number of IDs are missing from 1..N, so the IDs were overwritten.
#    The file is ordered by original ID, so row order restores the key.
# --------------------------------------------------------------------------
dup_ids = df["id"].duplicated(keep=False).sum()
note("Rows sharing an ID with another row (before fix)", dup_ids)

if df["id"].duplicated().any() or df["id"].isna().any():
    df.insert(1, "original_id", df["id"])
    df["id"] = pd.Series(np.arange(1, len(df) + 1), dtype="Int64")
    note("IDs reassigned from row order", (df["id"] != df["original_id"]).sum())

assert df["id"].is_unique, "IDs are still not unique"

# --------------------------------------------------------------------------
# 8. Age vs birthday
# --------------------------------------------------------------------------
valid_bd = df["birthday"].notna() & df["age"].notna()
ref_date, best_mismatch = infer_reference_date(
    df.loc[valid_bd, "birthday"], df.loc[valid_bd, "age"]
)
print(f"Inferred age reference date: {ref_date.date()} ({best_mismatch} rows still disagree)")

recalc_age = calc_age(df["birthday"], ref_date).astype("Int64")
age_wrong = df["age"].ne(recalc_age).fillna(False)
note("Age corrected from birthday", age_wrong.sum())
df.loc[age_wrong, "age"] = recalc_age[age_wrong]
df["age"] = df["age"].astype("Int64")

# --------------------------------------------------------------------------
# 9. Check-in / check-out / time in gym
# --------------------------------------------------------------------------
check_in = pd.to_timedelta(df["avg_time_check_in"], errors="coerce")
check_out = pd.to_timedelta(df["avg_time_check_out"], errors="coerce")
note("Unparseable check-in/out times", (check_in.isna() | check_out.isna()).sum())

minutes = (check_out - check_in).dt.total_seconds() / 60
minutes = minutes.where(minutes >= 0, minutes + 24 * 60)  # session past midnight
rounded_minutes = pd.Series(np.round(minutes), index=df.index).astype("Int64")

bad_minutes = (rounded_minutes != df["avg_time_in_gym"]).fillna(False)
note("avg_time_in_gym recomputed from check-in/out", bad_minutes.sum())
df.loc[bad_minutes, "avg_time_in_gym"] = rounded_minutes[bad_minutes]

# --------------------------------------------------------------------------
# 10. visit_per_week vs days_per_week
# --------------------------------------------------------------------------
days_count = df["days_per_week"].fillna("").apply(
    lambda s: len([d for d in s.split(",") if d.strip()])
)
bad_visits = (days_count != df["visit_per_week"]).fillna(False) & (days_count > 0)
note("visit_per_week corrected from days_per_week", bad_visits.sum())
df.loc[bad_visits, "visit_per_week"] = days_count[bad_visits].astype("Int64")

# --------------------------------------------------------------------------
# 11. Flag columns vs their detail columns
# --------------------------------------------------------------------------
checks = [
    ("attend_group_lesson", "fav_group_lesson"),
    ("drink_abo", "fav_drink"),
    ("personal_training", "name_personal_trainer"),
]
for flag, detail in checks:
    has_detail = df[detail].notna().astype("boolean")
    mismatch = (df[flag] != has_detail).fillna(False)
    note(f"{flag} inconsistent with {detail}", mismatch.sum())
    df.loc[mismatch, flag] = has_detail[mismatch]         # trust the detail
    df.loc[~df[flag].fillna(False), detail] = np.nan      # no flag -> no detail
    df[detail] = df[detail].fillna(NO_VALUE_LABEL)

# --------------------------------------------------------------------------
# 12. Range checks (flag and drop invalid rows)
# --------------------------------------------------------------------------
range_issue = (
    (~df["age"].between(1, 100)).fillna(True)
    | (~df["visit_per_week"].between(1, 7)).fillna(True)
    | (~df["avg_time_in_gym"].between(1, 24 * 60)).fillna(True)
)
note("Rows with out-of-range age/visits/duration (dropped)", range_issue.sum())
df = df[~range_issue].copy()

# Drop rows still missing key fields
key_cols = ["id", "gender", "birthday", "membership_type"]
before = len(df)
df = df.dropna(subset=key_cols)
note("Rows dropped for missing key fields", before - len(df))

# --------------------------------------------------------------------------
# 13. Final tidy-up and save
# --------------------------------------------------------------------------
df = df.sort_values("id").reset_index(drop=True)
df["birthday"] = df["birthday"].dt.strftime("%Y-%m-%d")
df.to_csv(OUTPUT_PATH, index=False)

print("\n=== CLEANING REPORT ===")
print(f"Rows in : {n_start}")
print(f"Rows out: {len(df)}")
for step, count in log:
    print(f"  {step:<62}{count:>6}")
missing_summary = df.isna().sum()[df.isna().sum() > 0]
print(f"\nMissing values left:\n{missing_summary.to_string() if not missing_summary.empty else 'none'}")
print(f"\nSaved to {OUTPUT_PATH}")