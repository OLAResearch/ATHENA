# MemoryATHENA arXiv source

This directory is a standalone arXiv-oriented derivative of the current paper
source in `paper/ICLR_submit.tex`. It keeps the existing claims, equations,
tables, appendix, and bibliography, while replacing the ICLR style dependency
with a conventional one-column article layout. The public resource links are
included near the abstract, following the project-page resource structure used
by the OLA Research template.

## Resources

- Code: <https://github.com/MJLee00/ATHENA>
- Models: <https://huggingface.co/collections/OLAResearchX/memoryathena>
- Dataset: <https://huggingface.co/datasets/OLAResearchX/MemoryATHENA>
- Project page source: <https://github.com/MJLee00/ATHENA/tree/main/web>

## Build

Run the build from this directory so that the relative figure and appendix
paths resolve correctly:

```bash
cd paper/arxiv
latexmk -pdf -interaction=nonstopmode MemoryATHENA_arxiv.tex
```

The current repository snapshot includes the framework figure but not all
derived result-figure PDFs referenced by the paper. Missing figures are
rendered as explicit source-snapshot notices by `\safeincludegraphics`; adding
the corresponding files under `figures/` makes them appear automatically.
