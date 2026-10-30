import argparse
import posixpath
import re
from pathlib import Path
from urllib.parse import urljoin

LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s#]+?)\.md(#[^)\s]*)?\)")


def site_url(root):
    match = re.search(
        r"^site_url:\s*(\S+)\s*$", (root / "mkdocs.yml").read_text(), re.M
    )
    if match is None:
        raise ValueError("mkdocs.yml must declare site_url")
    return match.group(1)


def absolute_notes(text, base_url, directory):
    def replace(match):
        path = posixpath.normpath(posixpath.join(directory, match.group(1)))
        target = "" if path == "index" else f"{path.removesuffix('/index')}/"
        return f"]({urljoin(base_url, target)}{match.group(2) or ''})"

    return LINK.sub(replace, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    root = Path.cwd()
    notes = root / "docs" / "release-notes" / f"{arguments.version}.md"
    arguments.output.write_text(
        absolute_notes(notes.read_text(), site_url(root), "release-notes")
    )


if __name__ == "__main__":
    main()
