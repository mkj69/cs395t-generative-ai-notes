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
3. Rename `learning-note-template.md` to `lecture-XX-short-topic.md`; every
   Markdown filename in the Logseq graph must be unique.
4. Fill the Markdown by hand, beginning with status `capturing`.
5. Move to `reconstructing` only after writing with the source closed.
6. Add a public index entry only when its in-progress status is unambiguous.

## Add a research note

1. Copy `notes/_template/` to `notes/<question-slug>/`.
2. Rename `research-note-template.md` to `<question-slug>.md` and similarly
   give any Markdown guide files unique, question-specific names.
3. Replace every bracketed prompt with your own reasoning.
4. Keep note-specific scripts, figures, and data provenance inside that folder.
5. Add a corresponding entry to `docs/data/notes.json` only when the note is
   ready to appear in the public index.
6. Create the rendered HTML page under `docs/notes/`.

## Publish from Logseq

The Markdown files under `course-notes/` and `notes/` are the canonical note
sources. After a change is pushed to `main`, GitHub Actions renders every note
marked `public: true` into the static `docs/` site and commits only the generated
site files. GitHub Pages continues to serve `docs/`, so the public site remains
dependency-free at runtime.

To preview the generated result locally, install `requirements-build.txt`, run
`python scripts/build_site.py`, and serve `docs/` as shown above.
