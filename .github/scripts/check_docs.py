"""The documentation site (docs/, zensical.toml) agrees with the package.

No network, no Julia. Checks that the nav and the page set match, that every internal link and
anchor resolves, that referenced assets exist, that every API name the pages document is actually
exported by `src/ARIS.jl`, and that no page carries placeholder text or an invented publication
URL.

usage (repo root):  python3 .github/scripts/check_docs.py
"""

from __future__ import annotations

import re
import sys
import tomllib
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CONFIG = ROOT / "zensical.toml"
SRC = ROOT / "src"

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.S | re.M)
LINK = re.compile(r"\[(?:[^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_SRC = re.compile(r"(?:src|href)=\"([^\"]+)\"")

failures: list[str] = []


def check(cond: object, msg: str) -> None:
    if not cond:
        failures.append(msg)


def pages() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(DOCS.glob("*.md"))}


def prose(text: str) -> str:
    """The page with fenced code removed, so examples are not scanned as prose."""
    return FENCE.sub("", text)


def slug(heading: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", heading)            # strip attr_list, e.g. { .cls }
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def anchors(text: str) -> set[str]:
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", prose(text), re.M)}


def nav_pages(nav: list) -> list[str]:
    out: list[str] = []
    for item in nav:
        for value in item.values():
            out += nav_pages(value) if isinstance(value, list) else [value]
    return out


def exported_names() -> set[str]:
    """Names exported by src/ARIS.jl, parsed statically from its `export` blocks."""
    text = (SRC / "ARIS.jl").read_text()
    text = re.sub(r"#[^\n]*", "", text)                  # drop comments
    names: set[str] = set()
    for m in re.finditer(r"\bexport\b(.*?)(?=\n\s*\n|\ninclude|\n@static|\nend\b)", text, re.S):
        for token in re.split(r"[,\s]+", m.group(1)):
            token = token.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_!]*", token or ""):
                names.add(token)
    return names


def main() -> int:
    config = tomllib.loads(CONFIG.read_text())["project"]
    docs = pages()

    # --- nav and page set are a bijection -------------------------------------------------
    nav = nav_pages(config["nav"])
    check(len(nav) == len(set(nav)), "a page appears more than once in the nav")
    for name in nav:
        check(name in docs, f"nav lists {name}, which does not exist in docs/")
    for name in docs:
        check(name in nav, f"docs/{name} exists but is not in the nav")

    # --- internal links and anchors resolve -----------------------------------------------
    page_anchors = {name: anchors(text) for name, text in docs.items()}
    for name, text in docs.items():
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            if not path:                                  # same-page anchor
                check(anchor in page_anchors[name],
                      f"docs/{name}: no heading for same-page anchor #{anchor}")
                continue
            if path.endswith(".md"):
                check(path in docs, f"docs/{name}: link to missing page {path}")
                if anchor and path in docs:
                    check(anchor in page_anchors[path],
                          f"docs/{name}: no heading for {path}#{anchor}")
            else:
                check((DOCS / path).exists(), f"docs/{name}: missing file {path}")

    # --- referenced assets exist ----------------------------------------------------------
    for name, text in docs.items():
        for src in IMG_SRC.findall(text):
            if not src.startswith(("http://", "https://")):
                check((DOCS / src).exists(), f"docs/{name}: missing asset {src}")
    for key in ("logo", "favicon"):
        ref = config.get("theme", {}).get(key)
        if ref:
            check((DOCS / ref).exists(), f"theme.{key} points at missing {ref}")
    for sheet in config.get("extra_css", []):
        check((DOCS / sheet).exists(), f"extra_css points at missing {sheet}")

    # the dark-mode logo is referenced from CSS, not from the config
    css = (DOCS / "stylesheets" / "extra.css").read_text()
    for ref in re.findall(r"url\(\"\.\./([^\"]+)\"\)", css):
        check((DOCS / ref).exists(), f"extra.css references missing asset {ref}")

    # --- documented API names are really exported -----------------------------------------
    exports = exported_names()
    check("is_estimate" in exports and "train!" in exports,
          "export parsing failed: expected names not found in src/ARIS.jl")
    # API pages head each entry with a code-formatted heading, e.g. "## `is_estimate`".
    allowed = exports | {
        "System", "SystemParameters",            # the module and its base type
        "GaussianMixture", "EMGMM",              # exported container types
        "ess", "bslice", "conditional", "marginal",
    }
    for name in (n for n in docs if n.startswith("api-")):
        for heading in re.findall(r"^#{2,3}\s+`([A-Za-z_][A-Za-z0-9_!]*)`\s*$", docs[name], re.M):
            check(heading in allowed,
                  f"docs/{name}: documents `{heading}`, which src/ARIS.jl does not export")

    # --- no placeholders, no invented publication identifiers ------------------------------
    banned = [
        (re.compile(r"\bTODO\b|\bTBD\b|\bFIXME\b|\bLorem ipsum\b", re.I), "placeholder text"),
        (re.compile(r"\bdoi\.org/|\barxiv\.org/abs/|\bjmlr\.org/", re.I),
         "a publication URL (the paper citation is not final)"),
        (re.compile(r"juliahub\.com"), "a registry reference (ARIS is not registered)"),
    ]
    for name, text in docs.items():
        body = prose(text)
        for pattern, what in banned:
            hit = pattern.search(body)
            check(hit is None, f"docs/{name}: contains {what}: {hit.group(0) if hit else ''}")

    # The install page must say outright that the package is not registered, since `Pkg.add` is
    # the first thing a Julia user will reach for.
    check("not registered" in docs["install.md"].lower(),
          "docs/install.md: does not state that the package is not in the General registry")

    # --- external links point only at this repository and known hosts ----------------------
    allowed_hosts = ("https://github.com/sisl/ARIS.jl", "https://julialang.org")
    for name, text in docs.items():
        for target in LINK.findall(text):
            if target.startswith("https://"):
                check(target.startswith(allowed_hosts),
                      f"docs/{name}: unexpected external link {target}")

    if failures:
        print(f"check_docs: {len(failures)} problem(s)\n", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"check_docs: OK ({len(docs)} pages, {len(exports)} exported names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
