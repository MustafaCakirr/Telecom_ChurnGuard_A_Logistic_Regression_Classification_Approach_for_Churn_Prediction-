import pandas as pd

# Remove outliers from the DataFrame using the IQR method.
def remove_outliers(df, columns):
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Outlier'ları filtrele
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
