import pandas as pd

def examine_csv():
    # Read the first few rows of the CSV file
    df = pd.read_csv('cuisines.csv', nrows=5)
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print("-" * 50)
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nFirst few rows:")
    print("-" * 50)
    print(df)
    
    print("\nData types of columns:")
    print("-" * 50)
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values in each column:")
    print("-" * 50)
    print(df.isnull().sum())

if __name__ == "__main__":
    examine_csv() 