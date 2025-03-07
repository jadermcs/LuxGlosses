import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

for model_name in ["gpt-4o-mini", "ft:gpt-4o-mini-2024-07-18:list:lod-translate:B7n1076x"]:
    count = 0
    correct = 0
    with open("test.jsonl") as fin:
        for line in fin.readlines():
            data = json.loads(line)["messages"]
            data[1]["content"] += " After reasoning give the answer in the last line."
            answer = data[-1]["content"]
            del data[-1]
            completion = client.chat.completions.create(
                model=model_name,
                messages=data
            )

            response = completion.choices[0].message.content.split("\n")[-1]
            if answer in response.lower():
                correct += 1
            count += 1
            if count >= 100:
                break

    print(correct/count)
