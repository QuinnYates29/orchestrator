"""Language detection and the string/comment masking every check relies on.

The original checks matched raw substrings, so `pprint(x)` and the string
literal `"use print(x) to debug"` both counted as debug prints. Nearly every
false positive in a line-based reviewer comes from the same place: matching
inside a string or a comment. So before any check runs, each line is split into

  code    - the executable part, with string *contents* blanked out
  comment - the comment part, if any

and checks match against whichever one they actually mean. `todo-comment` looks
only at comments; `debug-statement` looks only at code; `hardcoded-secret` is
the odd one out and looks at the raw line, because a secret lives inside the
string literal the other checks are trying to ignore.

Known limit: a diff hands us added lines out of context, so there is no way to
know whether the hunk began inside a multi-line string. Masking is therefore
per-line, and a `\"\"\"docstring\"\"\"` spanning several added lines is only
partly masked. Erring toward reporting is the right call for a reviewer - a
false positive costs a glance, a missed `except:` costs more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Language:
    name: str
    line_comments: tuple[str, ...] = ()
    quotes: tuple[str, ...] = ('"', "'")
    triple_quotes: tuple[str, ...] = ()
    block_comment: tuple[str, str] | None = None
    escape: str = "\\"


_C_STYLE = dict(line_comments=("//",), block_comment=("/*", "*/"))

PYTHON = Language("python", line_comments=("#",), triple_quotes=('"""', "'''"))
TEXT = Language("text")

LANGUAGES: dict[str, Language] = {
    "py": PYTHON, "pyi": PYTHON,
    "js": Language("javascript", quotes=('"', "'", "`"), **_C_STYLE),
    "jsx": Language("javascript", quotes=('"', "'", "`"), **_C_STYLE),
    "mjs": Language("javascript", quotes=('"', "'", "`"), **_C_STYLE),
    "cjs": Language("javascript", quotes=('"', "'", "`"), **_C_STYLE),
    "ts": Language("typescript", quotes=('"', "'", "`"), **_C_STYLE),
    "tsx": Language("typescript", quotes=('"', "'", "`"), **_C_STYLE),
    "go": Language("go", quotes=('"', "'", "`"), **_C_STYLE),
    "rs": Language("rust", **_C_STYLE),
    "java": Language("java", **_C_STYLE),
    "kt": Language("kotlin", **_C_STYLE),
    "swift": Language("swift", **_C_STYLE),
    "cs": Language("csharp", **_C_STYLE),
    "c": Language("c", **_C_STYLE), "h": Language("c", **_C_STYLE),
    "cc": Language("cpp", **_C_STYLE), "cpp": Language("cpp", **_C_STYLE),
    "cxx": Language("cpp", **_C_STYLE), "hpp": Language("cpp", **_C_STYLE),
    "php": Language("php", line_comments=("//", "#"), block_comment=("/*", "*/")),
    "rb": Language("ruby", line_comments=("#",)),
    "sh": Language("shell", line_comments=("#",)),
    "bash": Language("shell", line_comments=("#",)),
    "zsh": Language("shell", line_comments=("#",)),
    "pl": Language("perl", line_comments=("#",)),
    "lua": Language("lua", line_comments=("--",)),
    "sql": Language("sql", line_comments=("--",), block_comment=("/*", "*/")),
    "yaml": Language("yaml", line_comments=("#",)),
    "yml": Language("yaml", line_comments=("#",)),
    "toml": Language("toml", line_comments=("#",)),
    "ini": Language("ini", line_comments=("#", ";")),
    "cfg": Language("ini", line_comments=("#", ";")),
    "tf": Language("terraform", line_comments=("#", "//"), block_comment=("/*", "*/")),
    "dockerfile": Language("dockerfile", line_comments=("#",)),
    "makefile": Language("makefile", line_comments=("#",)),
    "css": Language("css", line_comments=(), block_comment=("/*", "*/")),
    "scss": Language("scss", **_C_STYLE),
    "html": Language("html", block_comment=("<!--", "-->")),
    "xml": Language("xml", block_comment=("<!--", "-->")),
    "md": Language("markdown", quotes=()),
    "rst": Language("rst", quotes=()),
    "txt": TEXT,
    "json": Language("json", line_comments=(), quotes=('"',)),
}

# Files with no extension that are still recognisable.
_BY_NAME: dict[str, Language] = {
    "dockerfile": LANGUAGES["dockerfile"],
    "makefile": LANGUAGES["makefile"],
    "gnumakefile": LANGUAGES["makefile"],
    "rakefile": LANGUAGES["rb"],
    "gemfile": LANGUAGES["rb"],
    "vagrantfile": LANGUAGES["rb"],
}


def detect(path: str) -> Language:
    """The language for `path`, or a generic text language if unrecognised.

    Never returns None: an unknown extension gets the universal checks rather
    than being skipped, which is the whole point of not being Python-only.
    """
    name = PurePosixPath(path).name.lower()
    if name in _BY_NAME:
        return _BY_NAME[name]
    suffix = PurePosixPath(name).suffix.lstrip(".")
    if suffix in LANGUAGES:
        return LANGUAGES[suffix]
    # "Dockerfile.prod", "Makefile.include"
    stem = PurePosixPath(name).stem.lower()
    return _BY_NAME.get(stem, TEXT)


@dataclass
class Line:
    """One added line, pre-split so checks match the part they mean."""
    number: int
    raw: str
    code: str = ""       # strings blanked, comment removed
    comment: str = ""    # comment body only, without the marker

    @property
    def stripped_code(self) -> str:
        return self.code.strip()


def split_code_and_comment(text: str, lang: Language) -> tuple[str, str]:
    """Split `text` into (code, comment).

    String *contents* are replaced with spaces rather than deleted, so every
    index in `code` still lines up with the same index in the raw line - a
    check that reports a column, or slices the line, stays correct.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    quotes = lang.quotes
    triples = lang.triple_quotes
    block = lang.block_comment

    while i < n:
        ch = text[i]

        # --- triple-quoted strings (python) ---
        matched_triple = next((t for t in triples if text.startswith(t, i)), None)
        if matched_triple:
            close = text.find(matched_triple, i + len(matched_triple))
            if close == -1:
                out.append(" " * (n - i))     # unterminated: blank to end of line
                return "".join(out), ""
            out.append(" " * (close + len(matched_triple) - i))
            i = close + len(matched_triple)
            continue

        # --- line comments ---
        marker = next((c for c in lang.line_comments if text.startswith(c, i)), None)
        if marker:
            return "".join(out), text[i + len(marker):]

        # --- block comments, opened and possibly closed on this line ---
        if block and text.startswith(block[0], i):
            close = text.find(block[1], i + len(block[0]))
            if close == -1:
                return "".join(out), text[i + len(block[0]):]
            body = text[i + len(block[0]):close]
            out.append(" " * (close + len(block[1]) - i))
            i = close + len(block[1])
            # A block comment mid-line is still a comment; keep its text so
            # todo-comment can see `/* TODO: fix */ x = 1`.
            rest_code, rest_comment = split_code_and_comment(text[i:], lang)
            return "".join(out) + rest_code, (body + " " + rest_comment).strip()

        # --- string literals ---
        if ch in quotes:
            j = i + 1
            while j < n:
                if lang.escape and text[j] == lang.escape:
                    j += 2
                    continue
                if text[j] == ch:
                    break
                j += 1
            end = min(j, n - 1)
            out.append(ch + " " * max(0, end - i - 1) + (ch if j < n else ""))
            i = j + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out), ""


def build_lines(added: list[tuple[int, str]], lang: Language) -> list[Line]:
    lines: list[Line] = []
    for number, raw in added:
        code, comment = split_code_and_comment(raw, lang)
        lines.append(Line(number=number, raw=raw, code=code, comment=comment))
    return lines
