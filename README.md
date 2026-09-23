# CS 395T — Foundations of Modern Generative AI

A living learning and research notebook for ideas developed while studying CS 395T.

This repository is intentionally separate from downloaded lecture PDFs and the
course-material archive. It separates manual lecture reconstruction from later
research synthesis, derivations, paper connections, unresolved tensions, and
research ideas.

## What is here now

- a dependency-free static website in `docs/`;
- a searchable note index with one clearly labeled learning note in progress;
- a manual, no-AI workflow for creating future learning notes;
- a reusable learning-note template under `course-notes/_template/`;
- a research-note template in both HTML and Markdown;
- a writing method that separates evidence, derivation, synthesis, and
  hypothesis;
- a proposed card for the Writing section of `mkj69.github.io`.

Lecture 09 currently contains structure and prompts only. No substantive course
explanation has been generated or represented as learned.

## Preview locally

From this repository:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Organizing principle

The repository uses two complementary units. **Learning notes** follow the
course sequence and record manual reconstruction. **Research notes** follow a
question and begin only after learning produces something worth investigating.
Each note keeps its prose, local code, figures, and provenance together.

See `ARCHITECTURE.md` for the full map.

## Add a learning note

1. Read `MANUAL_NOTE_WORKFLOW.md`.
2. Copy `course-notes/_template/` to `course-notes/lecture-XX-short-topic/`.
3. Fill the Markdown by hand, beginning with status `capturing`.
4. Move to `reconstructing` only after writing with the source closed.
5. Add a public index entry only when its in-progress status is unambiguous.

## Add a research note

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
