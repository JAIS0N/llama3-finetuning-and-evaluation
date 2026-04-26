from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="./merged-model",  # may need to change (see below)
    messages=[
        {"role": "user", "content": "Explain deep learning simply"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
