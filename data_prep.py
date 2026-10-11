from datasets import load_dataset
import pandas as pd

# Load the dataset with the available config
dataset = load_dataset(
    "TheBritishLibrary/blbooks",
    "1500_1899"
)

# Filter for Italian and Hungarian
italian_data = dataset['train'].filter(
    lambda x: 'Italian' in [x.get('Language_1'), x.get('Language_2'), x.get('Language_3'), x.get('Language_4')]
)

hungarian_data = dataset['train'].filter(
    lambda x: 'Hungarian' in [x.get('Language_1'), x.get('Language_2'), x.get('Language_3'), x.get('Language_4')]
)

# Convert to CSV
italian_data.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_books.csv", index=False)
hungarian_data.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_books.csv", index=False)

print(f"Saved {len(italian_data)} Italian books")
print(f"Saved {len(hungarian_data)} Hungarian books")

# Load the CSV files
italian_data = pd.read_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_books.csv")
hungarian_data = pd.read_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_books.csv")

# Check the first few columns in the data
print(italian_data.head())

# Check what raw_date looks like
print("Italian raw_date info:")
print(f"Data type: {italian_data['raw_date'].dtype}")
print(f"Sample values: {italian_data['raw_date'].head(10).tolist()}")
print(f"Unique values count: {italian_data['raw_date'].nunique()}")

# Convert raw_date to numeric
italian_data['raw_date'] = pd.to_numeric(italian_data['raw_date'], errors='coerce')
hungarian_data['raw_date'] = pd.to_numeric(hungarian_data['raw_date'], errors='coerce')

# Filter for 1850-1899 using raw_date column
italian_filtered = italian_data[
(italian_data['raw_date'] >= 1850) & (italian_data['raw_date'] <= 1899)
]

hungarian_filtered = hungarian_data[
    (hungarian_data['raw_date'] >= 1850) & (hungarian_data['raw_date'] <= 1899)
]

# Save the filtered data
italian_filtered.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_books_1850_1899.csv", index=False)
hungarian_filtered.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_books_1850_1899.csv", index=False)

print(f"Filtered Italian books (1850-1899): {len(italian_filtered)}")
print(f"Filtered Hungarian books (1850-1899): {len(hungarian_filtered)}")

#Sample for labeling
#sample_size = 50

#italian_sample = italian_filtered.sample(n = sample_size, random_state = 42)
#hungarian_sample = hungarian_filtered.sample(n = sample_size, random_state = 42)

# Save the samples
italian_sample.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_sample_for_labeling.csv", index=False)
hungarian_sample.to_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_sample_for_labeling.csv", index=False)