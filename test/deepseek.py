# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

from config import API_KEY2 as LOCAL_API_KEY

api_key = os.getenv("DEEPSEEK_API_KEY") or LOCAL_API_KEY

if not api_key:
    raise RuntimeError(
        "DeepSeek API key not provided. Set the DEEPSEEK_API_KEY environment "
        "variable or populate config.API_KEY."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是什么模型"},
    ],
    stream=False,
)

print(response.choices[0].message.content)
