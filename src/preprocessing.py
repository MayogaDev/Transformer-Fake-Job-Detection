import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load data from a CSV file.
    Args:
        filepath (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the dataframe by dropping unnecessary columns, filling NaN values, and combining text columns.
    Args:
        df (pandas.DataFrame): The input dataframe.
    Returns:
        pandas.DataFrame: The preprocessed dataframe.
    """
    df = df.drop(columns=['salary_range', 'job_id'])
    df = df.fillna(" ")
    df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
    df = df.drop(columns=['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type', 'required_education', 'industry', 'function'])
    df = df.reset_index(drop=True)
    return df

def split_data(df, test_size=0.5):
    """Split the dataframe into training, validation, and testing dataframes.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        test_size (float, optional): The proportion of the dataframe to include in the train/validation and testing subsets. Defaults to 0.2.
    
    Returns:
        tuple: A tuple containing the training/validation, and testing dataframes.
    """
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_val_df, test_df
