import pandas as pd

# Assuming df is your DataFrame
# Replace 'your_excel_file.xlsx' with your actual file name
df = pd.read_excel('train_cleaned.xlsx')

# Dictionary to store mappings in var
original_values_mapping = {}

# Iterate through columns and convert non-integer values to categorical type
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        # Create a mapping for the original values to codes
        mapping = dict(enumerate(df[col].astype('category').cat.categories))
        original_values_mapping[col] = mapping
        df[col] = df[col].astype('category')

# Convert categorical columns to codes
df_codes = df.apply(lambda x: x.cat.codes if isinstance(x.dtype, pd.CategoricalDtype) else x)

# Save the original values mapping to a JSON file
import json
with open('original_values_mapping.json', 'w') as file:
    json.dump(original_values_mapping, file)

df_codes.to_excel('train_cleaned_transformed.xlsx', index=False)
