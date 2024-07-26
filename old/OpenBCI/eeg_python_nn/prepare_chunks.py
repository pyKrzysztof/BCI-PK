import pandas as pd
import os


# Load the CSV file
file_path = 'dane/processed_linear_detrend.csv'
out_path = 'dane/chunks_processed_linear_detrend/'
df = pd.read_csv(file_path, sep="\t")
print("Loaded data.")

os.makedirs(out_path, exist_ok=True)

# Counters for filenames
L_counter = 1
R_counter = 1

# Variable to keep track of the current chunk
current_chunk = []

# Iterate through the DataFrame
for index, row in df.iterrows():
    last_col_value = row.iloc[-1]

    if last_col_value == 1:
        # Start a new chunk if 1 is encountered
        if current_chunk:
            current_chunk = []  # Reset the current chunk if it was not empty
        current_chunk = [row]
    elif last_col_value == 2:
        # Start a new chunk if 2 is encountered
        if current_chunk:
            current_chunk = []  # Reset the current chunk if it was not empty
        current_chunk = [row]
    elif last_col_value == -1 and current_chunk and current_chunk[0].iloc[-1] == 1:
        # End the chunk if -1 is encountered and it started with 1
        current_chunk.append(row)
        # Save the chunk to a file
        chunk_df = pd.DataFrame(current_chunk)
        chunk_df.to_csv(out_path + f'L{L_counter}.csv', header=False, index=False)
        print(f"L{L_counter} processed.")
        L_counter += 1
        current_chunk = []
    elif last_col_value == -2 and current_chunk and current_chunk[0].iloc[-1] == 2:
        # End the chunk if -2 is encountered and it started with 2
        current_chunk.append(row)
        # Save the chunk to a file
        chunk_df = pd.DataFrame(current_chunk)
        chunk_df.to_csv(out_path + f'R{R_counter}.csv', header=False, index=False)
        print(f"R{R_counter} processed.")
        R_counter += 1
        current_chunk = []
    elif current_chunk:
        # If in a chunk, add the row to the current chunk
        current_chunk.append(row)
