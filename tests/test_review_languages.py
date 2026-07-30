"""Language detection and the string/comment masking checks depend on."""
from __future__ import annotations

import pytest

from pipeline.review.languages import TEXT, detect, split_code_and_comment


@pytest.mark.parametrize("path,expected", [
    ("a.py", "python"),
    ("pkg/mod.py", "python"),
    ("A.TS", "typescript"),          # extension matching is case-insensitive
    ("app.jsx", "javascript"),
    ("main.go", "go"),
    ("lib.rs", "rust"),
    ("A.java", "java"),
    ("s.sh", "shell"),
    ("conf.yaml", "yaml"),
    ("conf.yml", "yaml"),
    ("notes.md", "markdown"),
    ("data.json", "json"),
    ("styles.css", "css"),
    ("Dockerfile", "dockerfile"),
    ("Makefile", "makefile"),
    ("deep/dir/Gemfile", "ruby"),
    ("Dockerfile.prod", "dockerfile"),
])
def test_detect(path, expected):
    assert detect(path).name == expected


def test_unknown_extension_falls_back_to_text_not_none():
    """An unrecognised file must still be reviewable by the universal checks."""
    assert detect("archive.weird") is TEXT
    assert detect("noextension") is TEXT


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _py(text):
    return split_code_and_comment(text, detect("a.py"))


def test_string_contents_are_blanked_but_the_line_keeps_its_length():
    """Indices must still line up with the raw line, so a check that reports a
    column or slices the text stays correct."""
    raw = 'msg = "use print(x) here"'
    code, _ = _py(raw)
    assert len(code) == len(raw)
    assert "print(" not in code
    assert code.startswith("msg = ")


def test_comment_is_split_off_and_returned_separately():
    code, comment = _py("x = 1  # TODO: fix")
    assert "TODO" not in code
    assert "TODO" in comment


def test_a_hash_inside_a_string_does_not_start_a_comment():
    code, comment = _py('colour = "#ff0000"  # red')
    assert comment.strip() == "red"
    assert "ff0000" not in code


def test_code_before_a_comment_survives():
    code, _ = _py("    print(x)  # debug")
    assert "print(" in code


def test_escaped_quote_does_not_end_the_string_early():
    code, _ = _py(r'a = "he said \"print(\" ok"')
    assert "print(" not in code


def test_triple_quoted_string_is_masked():
    code, _ = _py('doc = """contains print( and # hash"""')
    assert "print(" not in code
    assert code.startswith("doc = ")


def test_unterminated_triple_quote_masks_to_end_of_line():
    code, _ = _py('doc = """opening a docstring with print(')
    assert "print(" not in code


def test_c_style_line_comment():
    code, comment = split_code_and_comment("int x = 1; // TODO later", detect("a.c"))
    assert "TODO" not in code
    assert "TODO" in comment


def test_c_style_block_comment_inline():
    code, comment = split_code_and_comment("x = 1 /* TODO fix */ + 2", detect("a.c"))
    assert "TODO" not in code
    assert "TODO" in comment
    assert "+ 2" in code


def test_template_literal_is_a_string_in_javascript():
    code, _ = split_code_and_comment("const s = `console.log(x)`", detect("a.js"))
    assert "console.log(" not in code


def test_markdown_has_no_comment_syntax():
    lang = detect("a.md")
    assert not lang.line_comments and lang.block_comment is None
    code, comment = split_code_and_comment("A line with TODO in it", lang)
    assert comment == ""


def test_a_line_with_no_string_or_comment_passes_through_unchanged():
    raw = "    return value + 1"
    code, comment = _py(raw)
    assert code == raw
    assert comment == ""
