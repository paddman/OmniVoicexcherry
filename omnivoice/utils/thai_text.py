"""Conservative Thai text normalization for TTS datasets and inference.

The normalizer intentionally covers common structured text rather than trying to
rewrite every digit-like token.  It handles Thai digits, dates, times, money,
percentages, telephone numbers, IPv4 addresses, common abbreviations, and plain
numbers while preserving inline control tags such as ``[laughter]``.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Callable

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI_TO_ARABIC = str.maketrans(THAI_DIGITS, ARABIC_DIGITS)

_DIGIT_WORDS = (
    "ศูนย์",
    "หนึ่ง",
    "สอง",
    "สาม",
    "สี่",
    "ห้า",
    "หก",
    "เจ็ด",
    "แปด",
    "เก้า",
)
_POSITION_WORDS = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน")
_MONTHS = (
    "",
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
)

# Longest entries are replaced first to avoid partial matches.
_ABBREVIATIONS = {
    "พ.ศ.": "พุทธศักราช",
    "ค.ศ.": "คริสต์ศักราช",
    "กทม.": "กรุงเทพมหานคร",
    "รพ.": "โรงพยาบาล",
    "รร.": "โรงเรียน",
    "บจก.": "บริษัทจำกัด",
    "บมจ.": "บริษัทมหาชนจำกัด",
    "ดร.": "ดอกเตอร์",
    "นายกฯ": "นายกรัฐมนตรี",
}

_PROTECTED_RE = re.compile(r"\[[^\]\n]{1,120}\]|https?://\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DATE_RE = re.compile(r"(?<!\w)(?:วันที่\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?!\d)(?:\s*น\.?)?")
_MONEY_RE = re.compile(r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d{1,4})?)\s*(?:บาท|฿)")
_PERCENT_RE = re.compile(r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:%|เปอร์เซ็นต์)")
_IPV4_RE = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+66|0)(?:[\s-]?\d){8,10}(?!\d)")
_NUMBER_RE = re.compile(r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?![\w.])")


def _under_million_to_words(number: int) -> str:
    if not 0 <= number < 1_000_000:
        raise ValueError("number must be between 0 and 999999")
    if number == 0:
        return ""

    result: list[str] = []
    digits = [int(ch) for ch in f"{number:06d}"]
    for index, digit in enumerate(digits):
        if digit == 0:
            continue
        position = 5 - index
        if position == 1:  # tens
            if digit == 1:
                result.append("สิบ")
            elif digit == 2:
                result.append("ยี่สิบ")
            else:
                result.append(_DIGIT_WORDS[digit] + "สิบ")
        elif position == 0:  # units
            if digit == 1 and number > 1:
                result.append("เอ็ด")
            else:
                result.append(_DIGIT_WORDS[digit])
        else:
            result.append(_DIGIT_WORDS[digit] + _POSITION_WORDS[position])
    return "".join(result)


def thai_number_to_words(value: int | str) -> str:
    """Convert a non-decimal integer to standard Thai cardinal words."""
    if isinstance(value, str):
        cleaned = value.translate(THAI_TO_ARABIC).replace(",", "").strip()
        if not re.fullmatch(r"[+-]?\d+", cleaned):
            raise ValueError(f"not an integer: {value!r}")
        number = int(cleaned)
    else:
        number = int(value)

    if number == 0:
        return _DIGIT_WORDS[0]
    if number < 0:
        return "ลบ" + thai_number_to_words(-number)
    if number < 1_000_000:
        return _under_million_to_words(number)

    high, low = divmod(number, 1_000_000)
    prefix = thai_number_to_words(high) + "ล้าน"
    if low == 0:
        return prefix
    # Thai convention uses เอ็ด for a trailing one after a non-zero million group.
    if low == 1:
        return prefix + "เอ็ด"
    return prefix + _under_million_to_words(low)


def thai_digits_to_words(value: str, separator: str = " ") -> str:
    digits = value.translate(THAI_TO_ARABIC)
    return separator.join(_DIGIT_WORDS[int(ch)] for ch in digits if ch.isdigit())


def thai_decimal_to_words(value: str) -> str:
    cleaned = value.translate(THAI_TO_ARABIC).replace(",", "").strip()
    sign = ""
    if cleaned.startswith(("+", "-")):
        sign = "ลบ" if cleaned[0] == "-" else "บวก"
        cleaned = cleaned[1:]
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        raise ValueError(f"not a decimal number: {value!r}")
    integer, dot, fraction = cleaned.partition(".")
    result = sign + thai_number_to_words(int(integer))
    if dot:
        result += "จุด" + "".join(_DIGIT_WORDS[int(ch)] for ch in fraction)
    return result


def thai_money_to_words(value: str) -> str:
    cleaned = value.translate(THAI_TO_ARABIC).replace(",", "").strip()
    try:
        amount = Decimal(cleaned).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(f"not a money amount: {value!r}") from exc

    sign = "ลบ" if amount < 0 else ""
    amount = abs(amount)
    baht = int(amount)
    satang = int((amount - Decimal(baht)) * 100)
    words = sign + thai_number_to_words(baht) + "บาท"
    return words + ("ถ้วน" if satang == 0 else thai_number_to_words(satang) + "สตางค์")


def _replace_with_callback(pattern: re.Pattern[str], text: str, callback: Callable[[re.Match[str]], str]) -> str:
    return pattern.sub(callback, text)


def _protect(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = f"__OVPROTECTED_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    return _PROTECTED_RE.sub(repl, text), protected


def _restore(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _replace_abbreviations(text: str) -> str:
    for source in sorted(_ABBREVIATIONS, key=len, reverse=True):
        text = text.replace(source, _ABBREVIATIONS[source])
    return text


def normalize_thai_text(text: str) -> str:
    """Normalize common Thai TTS text while preserving control tags and URLs.

    The function is deterministic and deliberately conservative.  Raw text
    should still be retained beside the normalized transcript for auditability.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return ""

    text, protected = _protect(text)
    text = text.translate(THAI_TO_ARABIC)
    text = _replace_abbreviations(text)

    def date_repl(match: re.Match[str]) -> str:
        day, month, year = (int(match.group(i)) for i in range(1, 4))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match.group(0)
        era = "พุทธศักราช" if year >= 2400 else "คริสต์ศักราช" if year >= 1900 else "ปี"
        return f"วันที่{thai_number_to_words(day)} {_MONTHS[month]} {era}{thai_number_to_words(year)}"

    text = _replace_with_callback(_DATE_RE, text, date_repl)

    def time_repl(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second_text = match.group(3)
        result = thai_number_to_words(hour) + "นาฬิกา"
        if minute:
            result += thai_number_to_words(minute) + "นาที"
        if second_text and int(second_text):
            result += thai_number_to_words(int(second_text)) + "วินาที"
        return result

    text = _replace_with_callback(_TIME_RE, text, time_repl)
    text = _replace_with_callback(_MONEY_RE, text, lambda m: thai_money_to_words(m.group(1)))
    text = _replace_with_callback(
        _PERCENT_RE,
        text,
        lambda m: thai_decimal_to_words(m.group(1)) + "เปอร์เซ็นต์",
    )

    def ip_repl(match: re.Match[str]) -> str:
        octets = match.group(0).split(".")
        return " จุด ".join(thai_digits_to_words(part) for part in octets)

    text = _replace_with_callback(_IPV4_RE, text, ip_repl)

    def phone_repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw.startswith("+66"):
            digits = "66" + re.sub(r"\D", "", raw[3:])
            return "บวก " + thai_digits_to_words(digits)
        return thai_digits_to_words(re.sub(r"\D", "", raw))

    text = _replace_with_callback(_PHONE_RE, text, phone_repl)
    text = _replace_with_callback(_NUMBER_RE, text, lambda m: thai_decimal_to_words(m.group(1)))

    # Keep spacing predictable without touching punctuation inside protected text.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*([,;])\s*", r"\1 ", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    text = _restore(text.strip(), protected)
    return text


__all__ = [
    "normalize_thai_text",
    "thai_decimal_to_words",
    "thai_digits_to_words",
    "thai_money_to_words",
    "thai_number_to_words",
]
