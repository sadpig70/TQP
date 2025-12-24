# TQP arXiv Submission Package

**Category:** quant-ph  
**Title:** Temporal Quantum Processing for Efficient Quantum Simulation

---

## Package Contents

```
arxiv/
├── TQP_PRX.tex      # Main LaTeX source
├── references.bib   # BibTeX references
├── figures/
│   ├── benchmark_combined.png
│   └── temporal_scaling_combined.png
└── README.md        # This file
```

---

## Build Instructions

```bash
# 1. Compile LaTeX
pdflatex TQP_PRX.tex

# 2. Generate bibliography
bibtex TQP_PRX

# 3. Recompile for cross-references
pdflatex TQP_PRX.tex
pdflatex TQP_PRX.tex
```

---

## arXiv Submission

1. **Category:** quant-ph
2. **Archive to upload:** ZIP containing all files above
3. **License:** CC BY 4.0 (recommended for open access)

---

## Files to Copy

From `docs/prx/paper/`:

- `TQP_PRX.tex`
- `references.bib`

From `docs/prx/figures/`:

- `benchmark_combined.png`
- `temporal_scaling_combined.png`

---

## Notes

- LaTeX requires REVTeX4-2 package
- Install via: `tlmgr install revtex4-2`
- Figures must be in `figures/` subdirectory relative to .tex file
