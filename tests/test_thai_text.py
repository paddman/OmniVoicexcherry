from omnivoice.utils.thai_text import (
    normalize_thai_text,
    thai_money_to_words,
    thai_number_to_words,
)


def test_thai_number_rules():
    assert thai_number_to_words(0) == "ศูนย์"
    assert thai_number_to_words(11) == "สิบเอ็ด"
    assert thai_number_to_words(21) == "ยี่สิบเอ็ด"
    assert thai_number_to_words(101) == "หนึ่งร้อยเอ็ด"
    assert (
        thai_number_to_words(1_234_567)
        == "หนึ่งล้านสองแสนสามหมื่นสี่พันห้าร้อยหกสิบเจ็ด"
    )


def test_thai_money():
    assert (
        thai_money_to_words("1,234.50")
        == "หนึ่งพันสองร้อยสามสิบสี่บาทห้าสิบสตางค์"
    )
    assert thai_money_to_words("20") == "ยี่สิบบาทถ้วน"


def test_normalize_common_thai_tts_patterns():
    normalized = normalize_thai_text(
        "วันที่ 21/8/2569 เวลา 10:30 น. ราคา 1,234.50 บาท ลด 12.5%"
    )
    assert (
        "วันที่ยี่สิบเอ็ด สิงหาคม พุทธศักราชสองพันห้าร้อยหกสิบเก้า"
        in normalized
    )
    assert "สิบนาฬิกาสามสิบนาที" in normalized
    assert "หนึ่งพันสองร้อยสามสิบสี่บาทห้าสิบสตางค์" in normalized
    assert "สิบสองจุดห้าเปอร์เซ็นต์" in normalized


def test_normalizer_preserves_control_tags_and_urls():
    text = "[laughter] เปิด https://example.com/a1 แล้วมี ๒๕ คน"
    normalized = normalize_thai_text(text)
    assert "[laughter]" in normalized
    assert "https://example.com/a1" in normalized
    assert "ยี่สิบห้า" in normalized
