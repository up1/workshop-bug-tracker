# Get data from API
import os
import requests

# Fetch data from a public API
# http://localhost:8989/api/rest/issues?page_size=10&page=1&select=id,summary,description,severity
# Authentication required for APIs

response = requests.get(
    "http://localhost:8989/api/rest/issues?page_size=10&page=1&select=id,summary,description,severity",
    headers={
        "Authorization": os.getenv("API_TOKEN")
    },
    timeout=10
)
data = response.json()
# Process the data
processed_data = [item for item in data['issues']]
print(processed_data)
# Save processed data to a file
with open('processed_data.txt', 'w') as file:
    for item in processed_data:
        file.write(f"{item}\n")
# Print confirmation
print("Data processed and saved to processed_data.txt")