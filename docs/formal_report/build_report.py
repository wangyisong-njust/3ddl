#!/usr/bin/env python3
"""Build a formal LaTeX submission version of the 3DDL acceleration report."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FORMAL = DOCS / "formal_report"
SOURCE_MD = DOCS / "3ddl_acceleration_report.md"

TITLE = "Defect-Aware Acceleration Report for a 3D Semiconductor Inspection Pipeline"


def read_source() -> list[str]:
    return SOURCE_MD.read_text(encoding="utf-8").splitlines()


def extract_section(lines: list[str], heading: str, stop_heading: str) -> list[str]:
    start = lines.index(heading) + 1
    stop = lines.index(stop_heading)
    block = lines[start:stop]
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    return block


def strip_manual_toc(lines: list[str]) -> list[str]:
    start = lines.index("## Table of Contents")
    end = start + 1
    while end < len(lines) and lines[end].strip() != "\\newpage":
        end += 1
    if end < len(lines) and lines[end].strip() == "\\newpage":
        end += 1
    return lines[:start] + lines[end:]


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"^Chapter\s+\d+\s+", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
    return text.strip()


def preprocess_body(lines: list[str]) -> str:
    processed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("#"):
            hashes, title = line.split(" ", 1)
            title = normalize_heading_text(title)
            if title in {"References", "Publications from this Report"}:
                processed.append(f"{hashes} {title} {{.unnumbered}}")
            else:
                processed.append(f"{hashes} {title}")
            i += 1
            continue

        table_match = re.match(r"^\*\*Table\s+\d+\.\d+\s+(.+)\*\*$", line.strip())
        if table_match:
            processed.append(f"Table: {table_match.group(1).strip()}")
            i += 1
            continue

        image_match = re.match(r"^!\[(.*)\]\((.+)\)$", line.strip())
        if image_match:
            path = image_match.group(2).strip()
            if path.startswith("figures/"):
                path = "../" + path
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                fig_match = re.match(r"^\*\*Fig\.\s+\d+\.\d+\*\*\s+(.+)$", lines[j].strip())
                if fig_match:
                    caption = fig_match.group(1).strip()
                    processed.append(f"![{caption}]({path})")
                    i = j + 1
                    continue
            processed.append(f"![{image_match.group(1)}]({path})")
            i += 1
            continue

        if line.strip() == "\\newpage":
            processed.append("\\clearpage")
            i += 1
            continue

        processed.append(line)
        i += 1

    return "\n".join(processed).strip() + "\n"


def convert_markdown_fragment(markdown_text: str, output_tex: Path) -> None:
    cmd = [
        "pandoc",
        "--from=markdown+tex_math_dollars",
        "--to=latex",
        "--top-level-division=chapter",
        "--number-sections",
        "--output",
        str(output_tex),
    ]
    subprocess.run(cmd, input=markdown_text.encode("utf-8"), check=True)


def write_main_tex() -> Path:
    main_tex = FORMAL / "3ddl_acceleration_report_submission.tex"
    main_tex.write_text(
        r"""\documentclass[12pt,oneside]{report}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{fontspec}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{setspace}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{grffile}
\usepackage{etoolbox}
\usepackage{microtype}
\setmainfont{TeX Gyre Termes}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\captionsetup[figure]{labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false}
\captionsetup[table]{labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Defect-Aware Acceleration Report}
\fancyhead[R]{\nouppercase{\leftmark}}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}
\hypersetup{hidelinks}
\title{Defect-Aware Acceleration Report for a 3D Semiconductor Inspection Pipeline}
\date{April 2026}
\begin{document}
\begin{titlepage}
\centering
\vspace*{3cm}
{\Huge\bfseries Defect-Aware Acceleration Report for a 3D Semiconductor Inspection Pipeline\par}
\vspace{1.5cm}
{\Large Formal Report Draft\par}
\vfill
{\large April 2026\par}
\end{titlepage}
\pagenumbering{roman}
\chapter*{Acknowledgements}
\input{acknowledgements.tex}
\chapter*{Abstract}
\input{abstract.tex}
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage
\pagenumbering{arabic}
\input{body.tex}
\end{document}
""",
        encoding="utf-8",
    )
    return main_tex


def build_pdf(main_tex: Path) -> None:
    for _ in range(2):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                main_tex.name,
            ],
            cwd=FORMAL,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )


def main() -> None:
    FORMAL.mkdir(parents=True, exist_ok=True)
    lines = read_source()

    acknowledgements = extract_section(lines, "## Acknowledgements", "## Abstract")
    abstract = extract_section(lines, "## Abstract", "## Table of Contents")

    body_lines = strip_manual_toc(lines)
    title_index = body_lines.index("# Defect-Aware Acceleration Report for a 3D Semiconductor Inspection Pipeline")
    body_lines = body_lines[title_index + 1 :]
    # Drop the already-extracted front matter.
    ack_index = body_lines.index("## Acknowledgements")
    intro_index = body_lines.index("# Chapter 1 Introduction")
    body_lines = body_lines[:ack_index] + body_lines[intro_index:]

    body_md = preprocess_body(body_lines)

    ack_md = "\n".join(acknowledgements).strip() + "\n"
    abstract_md = "\n".join(abstract).strip() + "\n"

    convert_markdown_fragment(ack_md, FORMAL / "acknowledgements.tex")
    convert_markdown_fragment(abstract_md, FORMAL / "abstract.tex")
    convert_markdown_fragment(body_md, FORMAL / "body.tex")
    (FORMAL / "submission_source.md").write_text(body_md, encoding="utf-8")

    main_tex = write_main_tex()
    build_pdf(main_tex)

    print("Generated formal report assets:")
    for name in [
        "acknowledgements.tex",
        "abstract.tex",
        "body.tex",
        "submission_source.md",
        "3ddl_acceleration_report_submission.tex",
        "3ddl_acceleration_report_submission.pdf",
    ]:
        path = FORMAL / name
        print(path)


if __name__ == "__main__":
    main()
