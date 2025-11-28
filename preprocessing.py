"""Preprocessing utilities for the survey dataset.

Provides `load_and_preprocess` which reads `combined_survey_data.csv`,
cleans the `Value` column, one-hot encodes categoricals, scales numeric
columns, and returns both the original dataframe and a processed
numeric DataFrame suitable for clustering.
"""

import os
import sys
import pandas as pd
import numpy as np


def load_and_preprocess(input_path='combined_survey_data.csv', save_path='processed_dataset.csv', verbose=True):
    """Load combined CSV, clean, preprocess and return (df_raw, df_processed).

    - input_path: path to combined csv (relative to this file if not absolute)
    - save_path: where to write processed numeric CSV (relative to cwd)
    - returns: (df_raw, df_processed) where df_processed is a pandas DataFrame
      containing numeric features (NaNs filled with 0)
    """
    base_dir = os.path.dirname(__file__)
    full_input = input_path if os.path.isabs(input_path) else os.path.join(base_dir, input_path)
    if not os.path.exists(full_input):
        raise FileNotFoundError(f"Input file not found: {full_input}")

    df = pd.read_csv(full_input)
    if verbose:
        print(f"Loaded {full_input} — shape: {df.shape}")

    # CLEAN Value column
    if 'Value' in df.columns:
        df['Value'] = df['Value'].replace('***', pd.NA)
        df['Value'] = df['Value'].astype(str).str.strip()
        df['Value'] = df['Value'].str.replace('%', '', regex=False)
        df['Value'] = df['Value'].str.replace(',', '', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
    else:
        raise KeyError("Expected column 'Value' not found in input CSV")

    # Basic stats
    if verbose:
        print('\nSummary statistics for Value:')
        print(df['Value'].describe())

    # Prepare columns
    categorical_cols = [c for c in ['Group', 'Education Level', 'Year', 'Domain', 'Indicator'] if c in df.columns]
    numeric_cols = ['Value']

    # Import sklearn pieces lazily with helpful message
    try:
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
    except Exception:
        print('Missing scikit-learn. Install with: pip3 install scikit-learn')
        raise

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    scaler = StandardScaler()

    transformers = []
    if categorical_cols:
        transformers.append(('cat', ohe, categorical_cols))
    transformers.append(('num', scaler, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit-transform
    X = preprocessor.fit_transform(df)

    # Convert to DataFrame (ColumnTransformer with sparse=False returns ndarray)
    df_processed = pd.DataFrame(X)

    # Save processed dataset
    try:
        df_processed.to_csv(save_path, index=False)
        if verbose:
            print(f"Processed dataset saved to: {save_path}")
    except Exception as e:
        if verbose:
            print(f"Warning: failed to save processed dataset: {e}")

    # Fill NaNs (if any) for clustering
    df_processed = df_processed.fillna(0)

    return df, df_processed
