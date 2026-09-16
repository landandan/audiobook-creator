"""
Audiobook Creator - Language Configuration Module
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Centralized per-language configuration. All language-specific behaviour
(dialogue quote styles, chapter headings, punctuation sets, filename rules)
is driven by the BOOK_LANGUAGE environment variable ("en" default / "zh").
English behaviour is byte-for-byte identical to the original project.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

BOOK_LANGUAGE = os.environ.get("BOOK_LANGUAGE", "en").strip().lower()


def is_chinese() -> bool:
    """Whether the book being processed is a Chinese book."""
    return BOOK_LANGUAGE in ("zh", "zh-cn", "zh-tw", "chinese", "cn")


# ---------------------------------------------------------------------------
# Dialogue quotes
# ---------------------------------------------------------------------------

# Chinese dialogue is written with curly quotes “…”; web novels also commonly
# use 「…」/『…』 (Taiwan/Japan style) or plain ASCII straight quotes.
ZH_DIALOGUE_PATTERN = re.compile(r'“[^”]+”|「[^」]+」|『[^』]+』|"[^"]+"')
EN_DIALOGUE_PATTERN = re.compile(r'"[^"]+"')


def get_dialogue_pattern() -> "re.Pattern":
    """Return the regex used to locate quoted dialogue for the book language."""
    return ZH_DIALOGUE_PATTERN if is_chinese() else EN_DIALOGUE_PATTERN


def has_dialogue_in_line(line: str) -> bool:
    """Check whether a line contains quoted dialogue (language-aware)."""
    return bool(get_dialogue_pattern().search(line))


def normalize_chinese_quotes(text: str) -> str:
    """
    Normalize the various Chinese quote styles to the standard curly quotes “…”
    so downstream dialogue splitting only has to care about one style.
    「…」 and 『…』 are converted; “…” is kept as-is.
    """
    if not text:
        return text

    # 「…」 -> “…” ; 『…』 -> “…”
    text = re.sub(r'「([^」]*)」', r'“\1”', text)
    text = re.sub(r'『([^』]*)』', r'“\1”', text)
    return text


# ---------------------------------------------------------------------------
# Punctuation
# ---------------------------------------------------------------------------

# Full-width Chinese punctuation, used for "only punctuation" detection and
# for deciding whether a line already ends with sentence punctuation.
ZH_PUNCTUATION = '，。！？；：、…—～·“”‘’（）《》〈〉【】「」『』〔〕％‰'
ZH_SENTENCE_ENDINGS = ('。', '！', '？', '…', '：', '；', '”', '’', '」', '』')


# ---------------------------------------------------------------------------
# Chapter headings
# ---------------------------------------------------------------------------

# Matches: 第一章 / 第12章 / 第一千三百四十五章 / 第一回 / 第三卷 / 第二部 /
#          第5节 / 第10集 / 楔子 / 序章 / 序言 / 尾声 / 终章 / 番外 / 番外篇
_ZH_CHAPTER_NUM = r'[零〇一二三四五六七八九十百千万两\d]+'
ZH_CHAPTER_HEADING_PATTERN = re.compile(
    rf'^\s*(第{_ZH_CHAPTER_NUM}[章回卷部节集]|楔子|序章|序言|尾声|终章|番外篇?)'
)

# Max length for a line to still be considered a chapter heading in Chinese
# books (headings are short standalone lines, e.g. "第一章 陨落的天才").
ZH_CHAPTER_HEADING_MAX_LEN = 40

# Compact web-novel headings are often written without a separator, for
# example "第一章陨落的天才". Keep supporting that convention while rejecting
# short prose sentences such as "第一章的内容很长".
ZH_COMPACT_CHAPTER_TITLE_MAX_LEN = 12
ZH_CHAPTER_TITLE_SEPARATORS = ':：、.-—_（(【['
ZH_PROSE_CONTINUATION_PREFIXES = (
    '的', '是', '中', '里', '内', '讲', '写', '说',
    '他', '她', '它', '我', '你', '这', '那', '其', '本',
    '将', '把', '被', '让', '与', '和', '却', '也', '就',
    '都', '还', '又', '并', '而', '但', '从', '在', '对',
    '向', '给', '由', '为', '以', '于', '如', '当', '若',
    '因', '虽',
)

_ZH_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_ZH_UNITS = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def chinese_numeral_to_int(s: str):
    """
    Convert a Chinese numeral string (e.g. 十二 / 三百零五 / 一千二百) or an
    Arabic digit string to an int. Returns None if it cannot be parsed.
    """
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        return int(s)

    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _ZH_DIGITS:
            number = _ZH_DIGITS[ch]
        elif ch in _ZH_UNITS:
            unit = _ZH_UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                if number == 0:  # e.g. "十五" means 1*10 + 5
                    number = 1
                section += number * unit
            number = 0
        else:
            return None
    return total + section + number


def is_chinese_chapter_heading(text: str) -> bool:
    """
    Whether a line is a Chinese chapter heading such as "第一章 陨落的天才",
    "第十二回", "楔子", "番外篇" etc.
    """
    if not text:
        return False
    line = text.strip()
    if not line or len(line) > ZH_CHAPTER_HEADING_MAX_LEN:
        return False

    match = ZH_CHAPTER_HEADING_PATTERN.match(line)
    if not match:
        return False

    heading = match.group(1)
    # Bare section markers (楔子/序章/尾声/番外...) are always headings.
    if not heading.startswith('第'):
        return True

    # 第X章 style: validate the numeral between 第 and the suffix.
    numeral = heading[1:-1]
    if chinese_numeral_to_int(numeral) is None:
        return False

    # Prefer an explicit title separator. Compact headings without a separator
    # are supported only when short and when the remainder does not begin like
    # a prose continuation ("第一章的内容...", "第十二章他终于...").
    remainder = line[match.end():]
    if not remainder:
        return True
    if remainder[0].isspace() or remainder[0] in ZH_CHAPTER_TITLE_SEPARATORS:
        return True
    if remainder.startswith(ZH_PROSE_CONTINUATION_PREFIXES):
        return False
    if any(ch in remainder for ch in '。！？；'):
        return False

    return len(line) <= ZH_COMPACT_CHAPTER_TITLE_MAX_LEN


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------

# CJK Unified Ideographs (+ Extension A) range used to keep Chinese characters
# in generated filenames.
CJK_RANGE = r'一-鿿㐀-䶿'


def sanitize_filename_language_aware(text: str) -> str:
    """
    Language-aware filename sanitizer.

    English books keep the original ASCII-only behaviour. For Chinese books
    CJK characters are preserved; otherwise every chapter filename would be
    stripped to an empty/blank name and chapters would overwrite each other.
    """
    if not is_chinese():
        return text  # caller applies the original ASCII-only regex

    text = text.replace("'", '').replace('"', '').replace('/', ' ').replace('.', ' ')
    text = text.replace(':', '').replace('?', '').replace('\\', '').replace('|', '')
    text = text.replace('*', '').replace('<', '').replace('>', '').replace('&', 'and')
    # Chinese full-width punctuation that is invalid/awkward in filenames
    for ch in '：？｜＊＜＞“”‘’《》、，。！；':
        text = text.replace(ch, ' ')

    regex = rf"[^a-zA-Z0-9\-_./\s{CJK_RANGE}]"
    text = re.sub(regex, ' ', text, 0, re.MULTILINE)
    text = ' '.join(text.split())
    return text


# ---------------------------------------------------------------------------
# Main-content extraction markers (book_to_txt.py)
# ---------------------------------------------------------------------------

def get_default_content_markers():
    """Default (start, end) markers for extracting the main body of a book."""
    if is_chinese():
        return "楔子", "全书完"
    return "PROLOGUE", "ABOUT THE AUTHOR"


def get_chapter_marker_words():
    """
    Extra words treated as chapter/section markers when trimming the first and
    last lines of extracted main content. For Chinese books we return an empty
    list: single-character words like 章/卷 would match almost any line and
    would eat legitimate chapter headings (e.g. "第一章 陨落的天才").
    """
    if is_chinese():
        return []
    return ['chapter', 'part', 'book']
