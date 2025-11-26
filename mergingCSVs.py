"""Merge the provided survey CSV files into a single long-form CSV.

This script is tolerant of rows where the middle indicator column contains
commas (it joins extra columns into the indicator). It expects the source
files to live in the `Students drugs Addiction Dataset` folder next to
this script.
"""

import os
import csv
import sys
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing required Python packages: pandas and/or numpy.")
    print("Install them with: pip3 install pandas numpy")
    sys.exit(1)

# Directory containing the CSV files (relative to this script)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Students drugs Addiction Dataset')

col_map = {
    'ss2024_t4.7e.csv': ["Drug Usage Group", "Internet Habit or Activity", "Education Level", "Year", "Value"],
    'ss2024_t4.5e.csv': ["Drug Usage Group", "Peer Statement", "Education Level", "Year", "Value"],
    'ss2024_t4.6e.csv': ["Drug Usage Group", "Free Time Activity", "Education Level", "Year", "Value"],
    'ss2024_t4.4e.csv': ["Drug Usage Group", "Self Efficacy Statement", "Education Level", "Year", "Value"],
    'ss2024_t4.9e.csv': ["Drug Usage Group", "Alcohol/Tobacco Status", "Education Level", "Year", "Value"]
}


def tolerant_read_csv(path):
    """Read CSV and ensure each returned row has exactly 5 fields.

    If a row has more than 5 fields we join the middle fields into the
    indicator column. Short rows are padded with empty strings.
    """
    records = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return records
    for parts in rows[1:]:
        parts = [p.strip() for p in parts]
        if len(parts) > 5:
            record = [parts[0]]
            indicator = ','.join(parts[1:-3]).strip()
            record.append(indicator)
            record.extend(parts[-3:])
        else:
            record = parts[:]
            while len(record) < 5:
                record.append('')
        records.append(record)
    return records


def csv_to_longdf(fname, domain, indicator_col):
    base = os.path.basename(fname)
    if base not in col_map:
        raise KeyError(f"Unexpected file name: {base}")
    cols = col_map[base]
    fullpath = fname if os.path.isabs(fname) else os.path.join(DATA_DIR, base)
    if not os.path.exists(fullpath):
        raise FileNotFoundError(fullpath)
    recs = tolerant_read_csv(fullpath)
    df = pd.DataFrame(recs, columns=cols)
    # Standardize columns
    df['Domain'] = domain
    if 'Alcohol/Tobacco Status' in df.columns:
        conds = [
            df['Alcohol/Tobacco Status'].str.contains('Both tobacco and alcohol', case=False, na=False),
            df['Alcohol/Tobacco Status'].str.contains('User of alcohol but not tobacco', case=False, na=False),
            df['Alcohol/Tobacco Status'].str.contains('User of tobacco but not alcohol', case=False, na=False),
            df['Alcohol/Tobacco Status'].str.contains('Neither tobacco nor alcohol', case=False, na=False)
        ]
        choices = ["Both tobacco and alcohol user", "User of alcohol only", "User of tobacco only", "Neither tobacco nor alcohol"]
        df['Group'] = np.select(conds, choices, default=df['Drug Usage Group'])
        df['Indicator'] = 'Drug-taking status'
    else:
        df['Group'] = df['Drug Usage Group']
        df['Indicator'] = df[indicator_col]

    # Trim whitespace on key columns
    for c in ['Group', 'Education Level', 'Year', 'Indicator', 'Value']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df[['Group', 'Education Level', 'Year', 'Domain', 'Indicator', 'Value']]


domains_for = [
    ("ss2024_t4.7e.csv", 'Internet Habits', "Internet Habit or Activity"),
    ("ss2024_t4.5e.csv", 'Social and Peer Statements', "Peer Statement"),
    ("ss2024_t4.6e.csv", 'Free Time Activities', "Free Time Activity"),
    ("ss2024_t4.4e.csv", 'Self Efficacy', "Self Efficacy Statement"),
    ("ss2024_t4.9e.csv", 'Alcohol and Tobacco Use', "Alcohol/Tobacco Status")
]


if __name__ == '__main__':
    dfs = []
    for fname, domain, ind in domains_for:
        try:
            dfs.append(csv_to_longdf(fname, domain, ind))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    if not dfs:
        print('No data frames created. Exiting.')
    else:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(combined_df.shape)
        print(combined_df.head())
        combined_df.to_csv('combined_survey_data.csv', index=False)