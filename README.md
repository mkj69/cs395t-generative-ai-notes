# CS 395T — Foundations of Modern Generative AI

A living research notebook for ideas developed while studying CS 395T.

This repository is intentionally separate from downloaded lecture PDFs and the
course-material archive. It is for original synthesis: derivations, intuitions,
connections across papers, unresolved tensions, and research ideas.

## What is here now

- a dependency-free static website in `docs/`;
- a searchable note index that currently contains only a clearly labeled
  writing template;
- a research-note template in both HTML and Markdown;
- a writing method that separates evidence, derivation, synthesis, and
  hypothesis;
- a proposed card for the Writing section of `mkj69.github.io`.

No course notes have been invented or published in this initial scaffold.

## Preview locally

From this repository:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Organizing principle

The repository is organized around **research units**, not weeks or file types.
Each substantial note gets a folder that keeps its prose, local code, figures,
and data provenance together. Cross-note material is separated into research
threads, reusable experiments, and a source library.

See `ARCHITECTURE.md` for the full map.

## Add a note

1. Copy `notes/_template/` to `notes/<question-slug>/`.
2. Replace every bracketed prompt with your own reasoning.
3. Keep note-specific scripts, figures, and data provenance inside that folder.
4. Add a corresponding entry to `docs/data/notes.json` only when the note is
   ready to appear in the public index.
5. Create the rendered HTML page under `docs/notes/`.

The site is designed for GitHub Pages and does not require a build step.

The current first version stays dependency-free. If executable prose and
citations become the dominant workflow, the source layer can later move to
Quarto without changing the conceptual structure.
