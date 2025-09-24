# utils/sanitize.py
import os

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u2018": "'", "\u2019": "'",  # ‘ ’
}
NBSP = "\u00a0"

def ascii_clean(s: str) -> str:
    if s is None:
        return s
    for bad, good in SMART_QUOTES.items():
        s = s.replace(bad, good)
    s = s.replace(NBSP, " ")
    # 헤더 용 값은 ASCII만 허용
    s.encode("ascii")  # 실패 시 UnicodeEncodeError → 즉시 확인 가능
    return s

def get_ascii_env(key: str) -> str | None:
    v = os.getenv(key)
    if v is None:
        return None
    return ascii_clean(v)
