import pytest
from segmentation import segment_text, _split_paragraphs, _split_sentences


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def test_split_on_blank_line():
    text = "甲方应按时付款。\n\n乙方应按时交货。"
    paras = _split_paragraphs(text)
    assert len(paras) == 2


def test_split_on_single_newline_fallback():
    text = "甲方应按时付款。\n乙方应按时交货。"
    paras = _split_paragraphs(text)
    assert len(paras) == 2


def test_article_marker_triggers_new_paragraph():
    text = "第一条 总则\n本合同由甲乙双方签订。第二条 付款\n甲方应于每月五日前付款。"
    paras = _split_paragraphs(text)
    assert any("第一条" in p for p in paras)
    assert any("第二条" in p for p in paras)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_sentence_split_on_period():
    text = "甲方应按时付款。乙方应按时交货。双方均应遵守本合同。"
    sents = _split_sentences(text)
    assert len(sents) == 3


def test_sentence_split_on_mixed_punctuation():
    text = "甲方应按时付款！乙方应按时交货？双方均应遵守本合同；否则承担违约责任。"
    sents = _split_sentences(text)
    assert len(sents) == 4


# ---------------------------------------------------------------------------
# Mixed-mode segment_text
# ---------------------------------------------------------------------------

def test_short_paragraph_kept_intact():
    text = "甲方应按时付款。"  # well under 400 chars
    segs = segment_text(text, para_max_len=400, seg_max_len=250)
    assert segs == ["甲方应按时付款。"]


def test_long_paragraph_is_sub_split():
    # Build a paragraph clearly over 400 chars
    sentence = "甲方应当在收到乙方提交的付款申请后十个工作日内完成审核并予以付款。"  # ~32 chars
    text = sentence * 15  # ~480 chars, no paragraph break
    segs = segment_text(text, para_max_len=400, seg_max_len=250)
    assert len(segs) > 1
    for seg in segs:
        assert len(seg) <= 250


def test_no_segment_exceeds_max_len():
    # Stress test: many sentences packed into one paragraph
    sentence = "本合同依照中华人民共和国相关法律法规签订，双方应严格履行各自义务，不得无故违约。"
    text = sentence * 20
    segs = segment_text(text, para_max_len=400, seg_max_len=250)
    for seg in segs:
        assert len(seg) <= 250, f"Segment too long ({len(seg)}): {seg[:40]}…"


def test_multiple_paragraphs_mixed():
    short_para = "甲方为合同的委托方。"
    long_sentence = "乙方承诺在合同有效期内提供符合国家标准及行业规范要求的全套技术服务，并对服务质量承担完全责任。"
    long_para = long_sentence * 14  # > 400 chars

    text = short_para + "\n\n" + long_para
    segs = segment_text(text, para_max_len=400, seg_max_len=250)

    assert segs[0] == short_para
    for seg in segs[1:]:
        assert len(seg) <= 250


def test_utf8_and_gbk_encoding(tmp_path):
    from segmentation import segment_file
    content = "甲方应按时付款。\n\n乙方应按时交货。"

    utf8_file = tmp_path / "contract_utf8.txt"
    utf8_file.write_text(content, encoding='utf-8')
    segs_utf8 = segment_file(utf8_file)
    assert len(segs_utf8) == 2

    gbk_file = tmp_path / "contract_gbk.txt"
    gbk_file.write_bytes(content.encode('gbk'))
    segs_gbk = segment_file(gbk_file)
    assert len(segs_gbk) == 2


def test_empty_lines_ignored():
    text = "\n\n\n甲方应按时付款。\n\n\n"
    segs = segment_text(text)
    assert segs == ["甲方应按时付款。"]


def test_hard_split_when_no_punctuation():
    # Single run of characters with no sentence-ending or secondary punctuation
    text = "甲" * 500  # no punctuation at all
    segs = segment_text(text, para_max_len=400, seg_max_len=250)
    for seg in segs:
        assert len(seg) <= 250
