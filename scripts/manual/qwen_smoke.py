import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is required")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    completion = client.chat.completions.create(
        model=os.getenv("QWEN_MODEL", "qwen3-max"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give one sentence about position sizing."},
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
