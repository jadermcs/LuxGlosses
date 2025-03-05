from openai import OpenAI
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-4o-mini-2024-07-18"
)

file_id = response["id"]
print("Uploaded file ID:", file_id)

