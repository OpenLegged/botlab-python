"""Declaration data files: :func:`read_config` and the restricted YAML parser.

A project keeps its declaration values in ``config/*.yaml`` and references them
from ``controller.py``; the source file stays structure and wiring only::

    runtime = read_config("config/runtime.yaml")
    project(runtime["name"], control_hz=runtime["control_hz"], ...)

The parser here is deliberately small and dependency-free: Pyodide ships no
PyYAML and the declaration channel serves file text through the host rather
than the virtual filesystem. The accepted subset is the one BotLab's
TypeScript ``parseRestrictedYaml`` accepts (YAML 1.2 core schema) and is
pinned against it by a dual-end fixture corpus: block mappings and sequences,
flow collections, quoted scalars, block scalars, comments and blank lines.
Anchors, aliases, tags, merge keys, directives, multiple documents, non-string
mapping keys and non-finite numbers are rejected instead of interpreted.

A config file may carry one reserved top-level entry, ``_label``: the human
display name the workspace tree shows for the file. :func:`read_config`
validates and strips it, so declaration data never sees it.
"""

import math
import re

import botlab_host

__all__ = ["read_config"]

_CONFIG_DIRECTORY = "config"
_DISPLAY_LABEL_KEY = "_label"
_LABEL_LOCALES = frozenset({"en", "zh"})
_MAX_SAFE_INTEGER = 2 ** 53 - 1

_NULL_TOKENS = frozenset({"", "~", "null", "Null", "NULL"})
_TRUE_TOKENS = frozenset({"true", "True", "TRUE"})
_FALSE_TOKENS = frozenset({"false", "False", "FALSE"})
_DECIMAL_INT = re.compile(r"[-+]?[0-9]+\Z")
_HEXADECIMAL_INT = re.compile(r"[-+]?0x[0-9a-fA-F]+\Z")
_OCTAL_INT = re.compile(r"[-+]?0o[0-7]+\Z")
_FLOAT_NUMBER = re.compile(r"[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?\Z")
_NON_FINITE = re.compile(r"[-+]?\.(?:inf|Inf|INF|nan|NaN|NAN)\Z")
_QUOTE_PAIRS = {'"', "'"}
_FLOW_DELIMITERS = ",[]{}"


def _config_path(path):
    """The canonical config file reference: ``config/<name>.yaml``."""
    if not isinstance(path, str) or not path:
        raise ValueError("read_config path must be a non-empty string")
    if "\\" in path:
        raise ValueError(f'read_config path "{path}" must use "/" separators')
    segments = path.split("/")
    if len(segments) != 2 or segments[0] != _CONFIG_DIRECTORY or not segments[1]:
        raise ValueError(
            f'read_config path "{path}" must name one file inside '
            f'"{_CONFIG_DIRECTORY}/", like "config/runtime.yaml"'
        )
    name = segments[1]
    if not name.endswith(".yaml") or name.endswith(".preset.yaml") or name.startswith("."):
        raise ValueError(
            f'read_config path "{path}" must be a project config .yaml file; '
            "preset calibration keeps its root-level *.preset.yaml form"
        )
    return path


class _ConfigMapping(dict):
    """Plain mapping data with a source path for declaration error messages."""

    def __init__(self, data, source_path):
        super().__init__(data)
        self._source_path = source_path


def read_config(path):
    """Read one ``config/*.yaml`` declaration file as plain Python data.

    The returned structure is JSON-compatible (mappings, sequences, strings,
    numbers, booleans, null) so it flows straight into ``project()``,
    resource constructors and ``control_profile``. Anything the restricted
    parser does not accept raises ``ValueError`` naming the file and line.

    A top-level ``_label`` entry names the file for humans (the workspace
    tree shows it instead of the filename). It is display metadata, never
    declaration data, so it is stripped here — ``**read_config(...)`` whole-
    mapping unpacks stay valid with a ``_label`` present.
    """
    source = _config_path(path)
    text = botlab_host.read_project_file(source)
    if not isinstance(text, str):
        raise ValueError(f'read_config("{source}") needs the file text from the host')
    data = _strip_display_label(source, _parse_restricted_yaml(source, text))
    return _ConfigMapping(data, source) if isinstance(data, dict) else data


def _strip_display_label(path, document):
    """Pop the reserved ``_label`` display name; malformed values hard-fail.

    The accepted forms mirror parameter labels: a non-empty string, or a
    ``{"en": ..., "zh": ...}`` table with ``en`` required. The value is
    discarded here — only the host-side tree reads it, straight from the file.
    """
    if not isinstance(document, dict) or _DISPLAY_LABEL_KEY not in document:
        return document
    label = document.pop(_DISPLAY_LABEL_KEY)
    valid = (
        isinstance(label, str) and label.strip()
        or isinstance(label, dict)
        and set(label) <= _LABEL_LOCALES
        and "en" in label
        and all(
            isinstance(value, str) and value.strip() for value in label.values()
        )
    )
    if not valid:
        raise ValueError(
            f"{path}: _label must be a non-empty string or a "
            'mapping like {"en": "Runtime", "zh": "运行时"} with "en" required; '
            "it is display metadata and never reaches project code"
        )
    return document


# ---------------------------------------------------------------------------
# Restricted YAML subset parser
#
# Line-oriented recursive descent. ``_index`` is the 0-based index of the next
# raw line to examine; ``_Line.number`` is that line's 1-based number.
# ---------------------------------------------------------------------------


class _YamlError(ValueError):
    def __init__(self, path, line, message):
        super().__init__(f"{path}:{line}: {message}")


def _resolve_plain_scalar(token, path, line):
    """YAML 1.2 core-schema resolution for one plain (unquoted) scalar."""
    if token in _NULL_TOKENS:
        return None
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    if _NON_FINITE.match(token):
        raise _YamlError(path, line, f'"{token}" is not a finite number')
    if _DECIMAL_INT.match(token):
        return _yaml_int(int(token, 10))
    if _HEXADECIMAL_INT.match(token):
        return _yaml_int(int(token, 16))
    if _OCTAL_INT.match(token):
        return _yaml_int(int(token, 8))
    if _FLOAT_NUMBER.match(token):
        value = float(token)
        if not math.isfinite(value):
            raise _YamlError(path, line, f'"{token}" is not a finite number')
        # JSON has one number type: the TypeScript side reads the same digits
        # as a float64, so integrals normalize to integers on both sides.
        if value.is_integer() and abs(value) <= _MAX_SAFE_INTEGER:
            return int(value)
        return value
    return token


def _yaml_int(value):
    if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
        return value
    return float(value)


def _find_key_separator(content):
    """Column of the ':' that ends a plain mapping key, or None.

    A ':' only separates a key when it ends the token (end of line) or is
    followed by a space; a '#' after whitespace starts a comment and ends the
    search. Plain keys may contain ':' as long as no space follows it.
    """
    for index in range(len(content)):
        character = content[index]
        if character == "#" and index > 0 and content[index - 1] in " \t":
            return None
        if character == ":" and (
            index + 1 == len(content) or content[index + 1] in " \t"
        ):
            return index
    return None


def _quoted_span(text, quote):
    """Index of the closing quote of the scalar starting at text[0], or None."""
    index = 1
    while index < len(text):
        character = text[index]
        if quote == "'":
            if character == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                return index
        else:
            if character == "\\":
                index += 2
                continue
            if character == quote:
                return index
        index += 1
    return None


def _quoted_key_entry(content):
    """True when a quoted scalar followed by ':' opens a compact mapping."""
    if content[0] not in _QUOTE_PAIRS:
        return False
    end = _quoted_span(content, content[0])
    if end is None:
        return False
    rest = content[end + 1:]
    return rest.startswith(":") and (len(rest) == 1 or rest[1] in " \t")


_DQUOTED_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0",
    "a": "\a", "b": "\b", "f": "\f", "v": "\v", "e": "\x1b",
    " ": " ", "/": "/",
}


def _decode_quoted(text, path, line):
    """Decode a complete quoted token (quotes included); line breaks fold to spaces."""
    quote = text[0]
    if quote == "'":
        return _fold_line_breaks(text[1:-1].replace("''", "'"))
    raw = _fold_line_breaks(text[1:-1])
    output = []
    index = 0
    while index < len(raw):
        character = raw[index]
        if character != "\\":
            output.append(character)
            index += 1
            continue
        marker = raw[index + 1] if index + 1 < len(raw) else ""
        if marker in _DQUOTED_ESCAPES:
            output.append(_DQUOTED_ESCAPES[marker])
            index += 2
            continue
        if marker in ("x", "u", "U"):
            width = {"x": 2, "u": 4, "U": 8}[marker]
            digits = raw[index + 2: index + 2 + width]
            try:
                output.append(chr(int(digits, 16)))
            except ValueError:
                raise _YamlError(
                    path, line, f'invalid \\{marker} escape in a "..." scalar'
                ) from None
            index += 2 + width
            continue
        raise _YamlError(path, line, f'unsupported escape "\\{marker}" in a "..." scalar')
    return "".join(output)


def _fold_line_breaks(text):
    """Flow-scalar folding: one break becomes a space, blank lines become breaks."""
    if "\n" not in text:
        return text
    output = ""
    pending_breaks = 0
    for part in text.split("\n"):
        part = part.strip()
        if part == "":
            pending_breaks += 1
            continue
        if not output:
            output = part
        elif pending_breaks:
            output += "\n" * pending_breaks + part
        else:
            output += " " + part
        pending_breaks = 0
    return output


def _strip_inline_comment(text):
    """Cut a plain scalar at the '#' that starts a comment (whitespace before it)."""
    for index in range(len(text)):
        if text[index] == "#" and (index == 0 or text[index - 1] in " \t"):
            return text[:index]
    return text


class _Line:
    __slots__ = ("number", "indent", "content")

    def __init__(self, number, indent, content):
        self.number = number
        self.indent = indent
        self.content = content


class _Parser:
    def __init__(self, path, text):
        if text.startswith("﻿"):
            text = text[1:]
        self._path = path
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        # A trailing newline terminates the last line; it does not open a new
        # empty one (block scalar `|+` chomping counts real blank lines only).
        if normalized.endswith("\n"):
            normalized = normalized[:-1]
        self._lines = normalized.split("\n") if normalized else [""]
        self._index = 0

    def _error(self, line_number, message):
        raise _YamlError(self._path, line_number, message)

    # -- line plumbing ------------------------------------------------------

    def _scan(self, index):
        """(indent, content) with comments kept; blank lines yield empty content."""
        raw = self._lines[index]
        stripped = raw.lstrip(" ")
        if stripped.startswith("\t"):
            self._error(index + 1, "tabs are not allowed as indentation")
        return len(raw) - len(stripped), stripped.rstrip()

    def _next_content(self):
        """0-based index of the next non-blank non-comment line, or None."""
        index = self._index
        while index < len(self._lines):
            _, content = self._scan(index)
            if content and not content.startswith("#"):
                return index
            index += 1
        return None

    def _peek(self):
        index = self._next_content()
        if index is None:
            return None
        indent, content = self._scan(index)
        return _Line(index + 1, indent, content)

    def _take(self):
        line = self._peek()
        if line is not None:
            self._index = line.number
        return line

    @staticmethod
    def _is_sequence_entry(content):
        return content.startswith("- ") or content == "-"

    # -- document -------------------------------------------------------------

    def parse_document(self):
        first = self._take()
        if first is None:
            self._error(1, "the YAML document must not be empty")
        if first.content.startswith("%"):
            self._error(first.number, "YAML directives are not allowed")
        if first.content == "---":
            first = self._take()
            if first is None:
                self._error(1, "the YAML document must not be empty")
        if first.content in ("---", "..."):
            self._error(first.number, "a document marker here would start a second document")
        if self._is_sequence_entry(first.content):
            self._error(first.number, "the YAML document root must be a mapping")
        if first.content[0] in "{[":
            # A flow mapping at the root is a JSON document — the format pyodide
            # file-rule handlers emit (Pyodide ships no PyYAML), and the TS end
            # accepts it as YAML 1.2, so this end must read the same documents.
            root = self._parse_flow(first, first.content)
            if not isinstance(root, dict):
                self._error(first.number, "the YAML document root must be a mapping")
        else:
            root = self._parse_mapping(first)
        trailing = self._peek()
        if trailing is not None:
            if trailing.content == "...":
                self._index = trailing.number
                after = self._peek()
                if after is None:
                    return root
                trailing = after
            self._error(trailing.number, f'unexpected content "{trailing.content}" after the document')
        return root

    # -- block collections ------------------------------------------------------

    def _parse_mapping(self, first):
        entries = {}
        line = first
        while line is not None:
            if line.indent != first.indent:
                if line.indent > first.indent:
                    self._error(line.number, "unexpected indentation")
                return entries
            if line.content in ("---", "..."):
                return entries
            if self._is_sequence_entry(line.content):
                self._error(line.number, "a sequence entry cannot appear inside a mapping")
            self._index = line.number
            key, rest = self._split_entry(line)
            if key in entries:
                self._error(line.number, f'duplicate mapping key "{key}"')
            entries[key] = (
                self._nested_value(first.indent)
                if rest is None
                else self._inline_value(line, rest, first.indent)
            )
            line = self._peek()
        return entries

    def _parse_sequence(self, first):
        items = []
        line = first
        while line is not None:
            if line.indent != first.indent:
                if line.indent > first.indent:
                    self._error(line.number, "unexpected indentation")
                return items
            if not self._is_sequence_entry(line.content):
                return items
            self._index = line.number
            if line.content == "-":
                items.append(self._nested_value(first.indent, dashes_below=False))
                line = self._peek()
                continue
            rest = line.content[2:]
            offset = line.indent + 2
            while rest.startswith(" "):
                rest = rest[1:]
                offset += 1
            items.append(
                self._sequence_item(first.indent, _Line(line.number, offset, rest))
            )
            line = self._peek()
        return items

    def _sequence_item(self, seq_indent, item_line):
        """One `- item`: a compact mapping entry or an inline value."""
        content = item_line.content
        if not content or content.startswith("#"):
            return self._nested_value(seq_indent, dashes_below=False)
        if content[0] in "&*!":
            self._error(item_line.number, "anchors, aliases and tags are not allowed")
        if content[0] in "{[":
            return self._inline_value(item_line, content, seq_indent)
        if _quoted_key_entry(content) or _find_key_separator(content) is not None:
            return self._parse_mapping(item_line)
        return self._inline_value(item_line, content, seq_indent)

    def _nested_value(self, parent_indent, dashes_below=True):
        """The value under an empty key or dash.

        Under a mapping key the value may be a deeper block, a sequence at the
        key's own indent, or null. Under a dash, dashes at the same indent are
        siblings, so only a deeper block or null can follow.
        """
        child = self._peek()
        if child is None or child.indent < parent_indent:
            return None
        if child.indent == parent_indent:
            if dashes_below and self._is_sequence_entry(child.content):
                return self._parse_sequence(child)
            return None
        return self._parse_block(self._take())

    def _parse_block(self, line):
        if self._is_sequence_entry(line.content):
            return self._parse_sequence(line)
        return self._parse_mapping(line)

    # -- entries ----------------------------------------------------------------

    def _split_entry(self, line):
        """Split `key: value` into the string key and value text.

        The value text is None when the entry's value lives on the following
        lines (empty or comment-only rest).
        """
        content = line.content
        if content[0] in _QUOTE_PAIRS:
            end = _quoted_span(content, content[0])
            if end is None:
                self._error(line.number, f"unterminated {content[0]}-quoted mapping key")
            key = _decode_quoted(content[: end + 1], self._path, line.number)
            rest = content[end + 1:].lstrip(" \t")
            if not rest.startswith(":"):
                self._error(line.number, "a quoted mapping key must be followed by ':'")
            rest = rest[1:]
        else:
            if content[0] in "&*":
                self._error(line.number, "anchors and aliases are not allowed")
            if content[0] == "!":
                self._error(line.number, "explicit YAML tags are not allowed")
            split = _find_key_separator(content)
            if split is None:
                self._error(line.number, f'expected "key: value", got "{content}"')
            key = _resolve_plain_scalar(content[:split].rstrip(), self._path, line.number)
            if not isinstance(key, str):
                self._error(line.number, "mapping keys must be strings")
            if key == "<<":
                self._error(line.number, "YAML merge keys are not allowed")
            rest = content[split + 1:]
        rest = rest.lstrip(" \t")
        return key, (rest or None)

    def _inline_value(self, line, rest, continuation_indent):
        if rest[0] in "&*!":
            kind = {"&": "anchor", "*": "alias", "!": "tag"}[rest[0]]
            self._error(line.number, f"YAML {kind}s are not allowed")
        if rest[0] in "{[":
            return self._parse_flow(line, rest)
        if rest[0] in "|>":
            return self._parse_block_scalar(line, rest)
        if rest[0] in _QUOTE_PAIRS:
            return self._parse_quoted_value(line, rest)
        return self._parse_plain_value(line, rest, continuation_indent)

    # -- scalars ------------------------------------------------------------------

    def _parse_quoted_value(self, line, rest):
        quote = rest[0]
        end = _quoted_span(rest, quote)
        if end is not None:
            trailing = rest[end + 1:].lstrip(" ")
            if trailing and not trailing.startswith("#"):
                self._error(line.number, f'unexpected "{trailing}" after a quoted scalar')
            return _decode_quoted(rest[: end + 1], self._path, line.number)
        parts = [rest.rstrip()]
        index = line.number
        while index < len(self._lines):
            _, content = self._scan(index)
            parts.append(content)
            if content and (span := _quoted_span(content, quote)) is not None:
                self._index = index + 1
                token = "\n".join(parts[:-1]) + "\n" + content[: span + 1]
                return _decode_quoted(token, self._path, line.number)
            index += 1
        self._error(line.number, f"unterminated {quote}-quoted scalar")

    def _parse_plain_value(self, line, rest, continuation_indent):
        """A plain scalar, possibly folded across more-indented following lines."""
        first = _strip_inline_comment(rest).rstrip()
        if _find_key_separator(first) is not None:
            self._error(line.number, "nested mappings are not allowed in compact mappings")
        parts = [first]
        index = line.number
        blanks = 0
        while index < len(self._lines):
            _, content = self._scan(index)
            if not content:
                blanks += 1
                index += 1
                continue
            if content.startswith("#"):
                break
            indent = len(self._lines[index]) - len(self._lines[index].lstrip(" "))
            if indent <= continuation_indent:
                break
            if _find_key_separator(content) is not None or self._is_sequence_entry(content):
                self._error(index + 1, "a plain scalar cannot continue into a mapping entry")
            if blanks:
                parts.append("")
                blanks = 0
            parts.append(content)
            index += 1
        self._index = index
        return _resolve_plain_scalar(
            _fold_line_breaks("\n".join(parts)).strip(), self._path, line.number
        )

    def _parse_block_scalar(self, line, rest):
        header = _strip_inline_comment(rest).rstrip()
        folded = header[0] == ">"
        chomp = "clip"
        explicit_indent = None
        for character in header[1:]:
            if character == "-":
                chomp = "strip"
            elif character == "+":
                chomp = "keep"
            elif character.isdigit():
                explicit_indent = int(character)
            else:
                self._error(line.number, f'unsupported block scalar header "{header}"')
        content_indent = None if explicit_indent is None else line.indent + explicit_indent
        collected = []
        index = line.number
        while index < len(self._lines):
            raw = self._lines[index]
            stripped = raw.lstrip(" ")
            if not stripped:
                collected.append("")
                index += 1
                continue
            indent = len(raw) - len(stripped)
            if content_indent is None:
                if indent <= line.indent:
                    break
                content_indent = indent
            if indent < content_indent:
                break
            collected.append(raw[content_indent:])
            index += 1
        self._index = index
        if not any(collected):
            return "\n" * len(collected) if chomp == "keep" else ""
        trailing = 0
        for part in reversed(collected):
            if part == "":
                trailing += 1
            else:
                break
        core = collected[: len(collected) - trailing]
        body = _fold_block_lines(core) if folded else core
        text = "\n".join(body)
        if chomp == "strip":
            return text
        if chomp == "clip":
            return text + "\n"
        return text + "\n" * (1 + trailing)

    # -- flow collections ------------------------------------------------------------

    def _parse_flow(self, line, rest):
        scanner = _FlowScanner(self, line.number, rest)
        value = scanner.parse_value()
        scanner.expect_end()
        self._index = scanner.line_number
        return value


def _fold_block_lines(lines):
    """Folded block scalar content: adjacent lines join with spaces, breaks stay."""
    output = None
    breaks = 0
    for line in lines:
        if line == "":
            breaks += 1
            continue
        if output is None:
            output = line
        elif breaks:
            output += "\n" * breaks + line
        else:
            output += " " + line
        breaks = 0
    return [] if output is None else [output]


class _FlowScanner:
    """Character scanner for one flow collection, which may span lines.

    Multi-line *plain* scalars inside a flow collection are not part of the
    subset: they stop at end of line and the collection then fails on the
    unexpected text rather than guessing a join.
    """

    def __init__(self, parser, line_number, text):
        self._parser = parser
        self._path = parser._path
        self._total_lines = len(parser._lines)
        self.line_number = line_number
        self._text = text
        self._position = 0

    def _error(self, message):
        raise _YamlError(self._path, self.line_number, message)

    def _skip_spaces(self):
        while self._position < len(self._text) and self._text[self._position] in " \t":
            self._position += 1

    def _carry(self):
        """Move to the next raw line, stripped of its indentation."""
        if self.line_number >= self._total_lines:
            return False
        self.line_number += 1
        raw = self._parser._lines[self.line_number - 1]
        stripped = raw.lstrip(" ")
        self._text = "" if stripped.startswith("#") else stripped.rstrip()
        self._position = 0
        return True

    def _skip_whitespace(self):
        """Skip spaces and line breaks; stops on content or end of input."""
        while True:
            self._skip_spaces()
            if self._position < len(self._text):
                return
            if not self._carry():
                return

    def expect_end(self):
        """Only spaces or a comment may follow the collection on its last line."""
        self._skip_spaces()
        if self._position < len(self._text) and self._text[self._position] != "#":
            self._error(f'unexpected "{self._text[self._position:]}" after a flow collection')

    def parse_value(self):
        self._skip_whitespace()
        if self._position >= len(self._text):
            self._error("a flow collection is missing its closing bracket")
        character = self._text[self._position]
        if character == "{":
            return self._parse_map()
        if character == "[":
            return self._parse_seq()
        if character in _QUOTE_PAIRS:
            return self._parse_quoted()
        return self._parse_plain()

    def _parse_map(self):
        self._position += 1
        entries = {}
        while True:
            self._skip_whitespace()
            if self._position >= len(self._text):
                self._error("a flow mapping is missing its closing '}'")
            if self._text[self._position] == "}":
                self._position += 1
                return entries
            key = self.parse_value()
            if not isinstance(key, str):
                self._error("mapping keys must be strings")
            if key in entries:
                self._error(f'duplicate mapping key "{key}"')
            self._skip_whitespace()
            if self._position < len(self._text) and self._text[self._position] == ":":
                self._position += 1
                entries[key] = self.parse_value()
            else:
                entries[key] = None
            self._skip_whitespace()
            if self._position >= len(self._text):
                self._error("a flow mapping is missing its closing '}'")
            character = self._text[self._position]
            if character == "}":
                self._position += 1
                return entries
            if character != ",":
                self._error(f'expected "," or "}}", got "{character}"')
            self._position += 1

    def _parse_seq(self):
        self._position += 1
        items = []
        while True:
            self._skip_whitespace()
            if self._position >= len(self._text):
                self._error("a flow sequence is missing its closing ']'")
            if self._text[self._position] == "]":
                self._position += 1
                return items
            items.append(self.parse_value())
            self._skip_whitespace()
            if self._position >= len(self._text):
                self._error("a flow sequence is missing its closing ']'")
            character = self._text[self._position]
            if character == "]":
                self._position += 1
                return items
            if character != ",":
                self._error(f'expected "," or "]", got "{character}"')
            self._position += 1

    def _parse_quoted(self):
        quote = self._text[self._position]
        token = ""
        fragment_start = self._position
        self._position += 1
        while True:
            if self._position >= len(self._text):
                token += self._text[fragment_start:] + "\n"
                if not self._carry():
                    self._error(f"unterminated {quote}-quoted scalar")
                fragment_start = 0
                self._position = 0
                continue
            character = self._text[self._position]
            if (
                quote == "'"
                and character == "'"
                and self._position + 1 < len(self._text)
                and self._text[self._position + 1] == "'"
            ):
                self._position += 2
                continue
            if quote == '"' and character == "\\":
                self._position += 2
                continue
            if character == quote:
                token += self._text[fragment_start: self._position + 1]
                self._position += 1
                return _decode_quoted(token, self._path, self.line_number)
            self._position += 1

    def _parse_plain(self):
        start = self._position
        position = self._position
        end = None
        while position < len(self._text):
            character = self._text[position]
            if character in _FLOW_DELIMITERS:
                end = position
                break
            if character == ":" and (
                position + 1 >= len(self._text)
                or self._text[position + 1] in " \t" + _FLOW_DELIMITERS
            ):
                end = position
                break
            if character == "#" and position > start and self._text[position - 1] in " \t":
                end = position
                break
            position += 1
        if end is None:
            end = len(self._text)
        self._position = end
        token = self._text[start:end].strip()
        if not token:
            self._error("expected a value in a flow collection")
        return _resolve_plain_scalar(token, self._path, self.line_number)


def _parse_restricted_yaml(path, text):
    return _Parser(path, text).parse_document()
