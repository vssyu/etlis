import re
from pathlib import Path
from typing import List


# Primary split: after Chinese sentence-ending punctuation
_SENTENCE_END = re.compile(r'(?<=[。！？；])\s*')

# Secondary split: after clause-internal punctuation when a sentence is still too long
_SECONDARY_SPLIT = re.compile(r'(?<=[，、：])\s*')

# Paragraph-level article markers common in Chinese contracts
_ARTICLE_MARKER = re.compile(
    r'^[\s]*(?:第[零一二三四五六七八九十百千\d]+[条章节款项]|[\（(][一二三四五六七八九十\d]+[\）)]|\d+[\.、])',
    re.MULTILINE,
)


def _split_paragraphs(text: str) -> List[str]:
    """
    Split on blank lines first; fall back to single newlines.
    Also treats lines starting with article markers as new paragraph boundaries.
    """
    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Insert an extra newline before article markers so they become paragraph heads
    text = _ARTICLE_MARKER.sub(lambda m: '\n' + m.group(), text)

    # Prefer blank-line splitting; if the whole doc is one block, use single newlines
    blocks = re.split(r'\n{2,}', text)
    if len(blocks) == 1:
        blocks = text.split('\n')

    return [b.strip() for b in blocks if b.strip()]


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_to_segments(sentences: List[str], max_len: int) -> List[str]:
    """
    Greedily concatenate sentences until the next one would push the segment
    past max_len.  Sentences that individually exceed max_len are sub-split
    on secondary punctuation; if still too long they are hard-split.
    """
    segments: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            segments.append(current.strip())
            current = ""

    def hard_split(text: str) -> List[str]:
        """Force-split text into chunks of at most max_len characters."""
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]

    def add_chunk(chunk: str):
        nonlocal current
        if len(chunk) > max_len:
            # Try secondary punctuation first
            sub_parts = _SECONDARY_SPLIT.split(chunk)
            for part in sub_parts:
                if len(part) > max_len:
                    flush()
                    segments.extend(hard_split(part))
                elif len(current) + len(part) <= max_len:
                    current += part
                else:
                    flush()
                    current = part
        elif len(current) + len(chunk) <= max_len:
            current += chunk
        else:
            flush()
            current = chunk

    for sent in sentences:
        add_chunk(sent)

    flush()
    return segments


def segment_text(
    text: str,
    para_max_len: int = 400,
    seg_max_len: int = 250,
) -> List[str]:
    """
    Mixed-mode segmentation for Chinese contract text.

    Paragraphs at or below para_max_len characters are kept intact.
    Longer paragraphs are split on sentence-ending punctuation and then
    merged greedily so that no output segment exceeds seg_max_len characters.

    Args:
        text: Raw contract text.
        para_max_len: Character threshold above which a paragraph is sub-split.
        seg_max_len: Maximum characters allowed in any output segment.

    Returns:
        Ordered list of text segments.
    """
    paragraphs = _split_paragraphs(text)
    segments: List[str] = []

    for para in paragraphs:
        if len(para) <= para_max_len:
            segments.append(para)
        else:
            sentences = _split_sentences(para)
            segments.extend(_merge_to_segments(sentences, seg_max_len))

    return segments


def segment_file(
    file_path: str | Path,
    encoding: str = 'utf-8',
    para_max_len: int = 400,
    seg_max_len: int = 250,
) -> List[str]:
    """Load a .txt file and return its segments."""
    path = Path(file_path)
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        text = path.read_text(encoding='gbk')
    return segment_text(text, para_max_len, seg_max_len)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python segmentation.py <contract.txt> [para_max_len] [seg_max_len]')
        sys.exit(1)

    para_max = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    seg_max = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    segs = segment_file(sys.argv[1], para_max_len=para_max, seg_max_len=seg_max)

    for idx, seg in enumerate(segs, 1):
        preview = seg[:80] + ('…' if len(seg) > 80 else '')
        print(f'[{idx:03d}] ({len(seg):>4d} chars) {preview}')

    print(f'\nTotal segments: {len(segs)}')
