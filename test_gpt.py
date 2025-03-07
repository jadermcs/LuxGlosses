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
            answer = data[-1]["content"]
            del data[-1]
            completion = client.chat.completions.create(
                model=model_name,
                messages=data
            )

            print(completion.choices[0].message)
            print(answer)
            if answer in completion.choices[0].message.content:
                correct += 1
            count += 1
            break

    print(correct/count)
