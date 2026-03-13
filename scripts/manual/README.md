# Manual Smoke Scripts

These scripts are for ad-hoc provider and connectivity checks. They are not part of `pytest`.

Examples:

```bash
python scripts/manual/binance_public_smoke.py
OPENAI_API_KEY=... python scripts/manual/openai_smoke.py
DEEPSEEK_API_KEY=... python scripts/manual/deepseek_smoke.py
QWEN_API_KEY=... python scripts/manual/qwen_smoke.py
```

Notes:
- Keep credentials in environment variables, not in source files.
- Expect these scripts to make real outbound API calls.
