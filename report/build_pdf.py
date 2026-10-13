# /// script
# requires-python = ">=3.12"
# dependencies = ["markdown"]
# ///
# Run with: uv run report/build_pdf.py
# Without uv, install the `markdown` package first.
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import markdown


REPORT_NAME = "final_report"
TIMEOUT_SECONDS = 60
CHROME_COMMANDS = ["google-chrome", "chromium", "chromium-browser", "msedge"]
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]
# Same look as the Assignment 1 report. Code is sized relative to the text
# around it: 0.864em is 9.5pt in the 11pt body and scales down inside tables.
STYLE = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.45;
       max-width: 17cm; margin: 2cm auto; color: #222; }
h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: .2em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
th { background: #f2f2f2; }
code { background: #f4f4f4; padding: 1px 4px; font-size: 0.864em; }
pre { background: #f4f4f4; padding: 8px; font-size: 9pt; }
pre code { padding: 0; font-size: inherit; }
img { max-width: 100%; }
em { color: #555; }
hr { border: 0; border-top: 1px solid #ccc; }
/* Keep headings and captions on the same page as their section, figure or table. */
h1, h2, h3, p:has(> img):has(+ p > em), p:has(+ table) { break-after: avoid; }
table { break-inside: avoid; }
"""


def find_chrome():
    for command in CHROME_COMMANDS:
        path = shutil.which(command)
        if path:
            return path
    for path in CHROME_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Could not find Chrome, Chromium or Edge")


def build_html(markdown_path, html_path):
    text = markdown_path.read_text(encoding="utf-8")
    title = next(
        line[2:].strip() for line in text.splitlines() if line.startswith("# ")
    )
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html_path.write_text(
        "<!doctype html>\n"
        f'<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>{title}</title>\n'
        f"<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n",
        encoding="utf-8",
    )


def pdf_size(pdf_path):
    return pdf_path.stat().st_size if pdf_path.exists() else 0


def print_to_pdf(html_path, pdf_path):
    pdf_path.unlink(missing_ok=True)
    # A separate profile keeps the build independent of an open browser window.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile_directory:
        process = subprocess.Popen(
            [
                find_chrome(),
                "--headless",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile_directory}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Chrome can stay open after printing, so wait for the file, not the exit.
        deadline = time.monotonic() + TIMEOUT_SECONDS
        previous_size = 0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
            size = pdf_size(pdf_path)
            if size > 0 and size == previous_size:
                break
            previous_size = size
        if process.poll() is None:
            process.terminate()
            process.wait()
    if pdf_size(pdf_path) == 0:
        raise RuntimeError("Chrome did not write the PDF")


def main():
    report_directory = Path(__file__).resolve().parent
    markdown_path = report_directory / f"{REPORT_NAME}.md"
    # The HTML stays next to the Markdown so the relative image paths resolve.
    html_path = report_directory / f"{REPORT_NAME}.html"
    pdf_path = report_directory / f"{REPORT_NAME}.pdf"

    build_html(markdown_path, html_path)
    print(f"HTML saved to: {html_path}")
    print_to_pdf(html_path, pdf_path)
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
