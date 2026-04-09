# Formal Report Build

This directory contains the LaTeX submission build for the 3DDL acceleration report.

## Files

- `build_report.py`: preprocesses the Markdown source, converts it to LaTeX, and builds a PDF.
- `submission_source.md`: generated intermediate Markdown with auto-number-friendly headings and captions.
- `acknowledgements.tex`, `abstract.tex`, `body.tex`: generated LaTeX fragments.
- `3ddl_acceleration_report_submission.tex`: generated main LaTeX file.
- `3ddl_acceleration_report_submission.pdf`: generated PDF output.

## Build

Run:

```bash
cd /home/kaixin/yisong/3ddl
python3 docs/formal_report/build_report.py
```

The build uses `pandoc` and `xelatex`. The generated PDF includes:

- title page
- acknowledgements
- abstract
- table of contents
- list of figures
- list of tables
- chapter numbering
- unified figure/table captions
- header and footer styling
