import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "You are a professional investor"},
            {"role": "user", "content": "Describe one risk rule for crypto futures trading."},
        ],
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
