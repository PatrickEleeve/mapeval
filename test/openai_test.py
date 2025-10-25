from openai import OpenAI
from config import API_KEY

client = OpenAI(api_key=API_KEY)
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": (
                "You are a professional cryptocurrency perpetual futures portfolio manager. "
    "You may trade the following contracts: {symbol_list}.\n\n"
    "**Objective**\n"
    "- Operate within a leverage band of +/-{leverage}x while targeting strong risk-adjusted returns "
    "and controlled drawdowns.\n\n"
    "**Available tools**\n"
    "- get_historical_prices(symbol, end_date, bars): retrieve the latest price history window.\n"
    "- calculate_moving_average(symbol, end_date, window_size): compute moving averages.\n"
    "- calculate_volatility(symbol, end_date, window_size): estimate historical volatility.\n\n"
    "**Guidelines**\n"
    "1. Evaluate recent price action and volatility.\n"
    "2. Decide the desired leverage exposure for each contract; positive values are long, "
    "negative values are short.\n"
    "3. Ensure the absolute value of every exposure stays within +/-{leverage}x and the sum "
    "of absolute exposures does not exceed {leverage}.\n\n"
    "**Output format**\n"
    'Respond strictly in JSON: {{"reasoning": "brief analysis ...", "exposure": {{"BTCUSDT": <float>, "ETHUSDT": <float>}}}}. '
    "Numbers represent leverage in multiples of account equity."
            ),
        }
    ],
)
print(response.choices[0].message.content)
