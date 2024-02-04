import csv

# Data for creating the CSV
data = [
    ["status", "followers", "friends", "fav", "account_age", "pic","label"],
    [6825, 100, 100, 20, 5, 10],
    # Add more data rows as needed
]

# Define the CSV file path
csv_file_path = 'data.csv'

# Write the data to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data)

print(f"CSV file '{csv_file_path}' created successfully.")
