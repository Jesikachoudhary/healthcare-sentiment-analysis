import pandas as pd

# Load the dataset
public_df = pd.read_csv("publicdataset.tsv", sep="\t")

# Check the columns in the dataset
print(public_df.columns)

# Check for the presence of 'review' and 'rating' or any similar columns
if 'drugName' in public_df.columns and 'usefulCount' in public_df.columns:
    # You can create a rating column based on the 'usefulCount'
    public_df["rating"] = public_df["usefulCount"].apply(lambda x: 1 if x <= 5 else (5 if x > 20 else 3))

    # Sentiment Column based on the rating
    public_df["Sentiment"] = public_df["rating"].apply(lambda x: 1 if x > 3 else 0)

    # Drop rows with missing values in the necessary columns ('review', 'rating')
    public_df = public_df.dropna(subset=["drugName", "rating"])

    # Add a new column for review length (if review text exists)
    if 'review' in public_df.columns:
        public_df["Length_of_Review"] = public_df["review"].apply(lambda x: len(str(x).split()))
    
    # Print the updated DataFrame
    print(public_df.head())
else:
    print("The necessary columns are missing from the dataset.")

