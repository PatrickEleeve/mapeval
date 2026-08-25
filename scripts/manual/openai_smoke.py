import os

from openai import OpenAI


PROMPT = """
You are a professional cryptocurrency perpetual futures portfolio manager.
Explain, in one short paragraph, how you would approach risk control for BTCUSDT and ETHUSDT.
""".strip()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5"),
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
