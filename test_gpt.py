import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


with open("test.json") as fin:
    for line in fin.readlines():
        data = json.loads(line)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=data
        )

        print(completion.choices[0].message)
        break

with open("test.json") as fin:
    for line in fin.readlines():
        data = json.loads(line)
        completion = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:list:letz-semantics:B7kHXZud",
            messages=data
        )

        print(completion.choices[0].message)
        break
