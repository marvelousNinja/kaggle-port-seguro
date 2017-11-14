import pandas as pd

def get_empty_columns_and_frequencies(df, empty_value):
    columns_with_empty_value = []
    empty_frequencies = []
    for column in df.columns.values:
        vals = df[column].values
        if empty_value in vals:
            columns_with_empty_value.append(column)
            empty_frequencies.append(len(vals[vals == empty_value]) / float(len(vals)))

    return pd.DataFrame({ 'Frequency': empty_frequencies }, index=columns_with_empty_value)

def get_unique_values_for_each_column(df):
    columns = df.columns.values
    unique_values = []

    for column in columns:
        unique_values.append(df[column].sort_values().unique())

    return pd.DataFrame({ 'Values': unique_values }, index=columns)
