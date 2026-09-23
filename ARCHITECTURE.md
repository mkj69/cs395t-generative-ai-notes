# Notebook architecture

## The central decision

Do not organize the public notebook by lecture number, and do not split all
Markdown, Python, figures, and references into unrelated format silos.

The primary unit is a **research question**. Its prose and supporting artifacts
stay close together:

```text
notes/<question-slug>/
├── note.md              # the argument and current conclusion
├── code/                # scripts or notebooks used only by this note
├── figures/             # explanatory and generated figures
└── data/README.md       # provenance; large/raw data stays external
```

## Four public surfaces

1. **Research threads** connect several notes and experiments around a larger
   question. A thread is a map, not another essay.
2. **Research notes** develop one question through intuition, derivation,
   evidence, connections, and open problems.
3. **Experiments** make a claim reproducible. They record code, configuration,
   environment, expected output, and interpretation—including negative
   results.
4. **Source library** records papers and other materials actually used. It
   stores citations, claim ledgers, and reading trails, but not downloaded PDFs.

## Repository map

```text
cs395t-generative-ai-notes/
├── notes/                 # question-centered research units
│   └── _template/
├── research/              # cross-note threads and open-question index
├── experiments/           # reusable or multi-note computational work
├── library/               # bibliography, paper metadata, reading records
├── src/                   # reusable code shared by several experiments
├── docs/                  # published GitHub Pages site
└── integration/           # proposed personal-homepage entry
```

## When code belongs where

- Put code in `notes/<slug>/code/` when it exists to explain or test that one
  note.
- Put a self-contained investigation in `experiments/<slug>/` when it has its
  own question, configuration, and result record or supports multiple notes.
- Move code into `src/` only after at least two research units genuinely reuse
  it.

## Source policy

- `library/references.bib` is the canonical citation database.
- `library/papers.yml` is the public-facing metadata and claim index.
- `library/reading-notes/` contains short, source-specific records of what was
  actually read and used.
- PDFs, slide decks, and large datasets stay outside the repository. Record a
  stable URL, DOI, license, checksum, or acquisition note instead.

## Publishing flow

```text
seed question
    ↓
private reconstruction + local experiment
    ↓
evidence and citation check
    ↓
reviewed research note
    ↓
public index + relevant research thread
```

The website should make these states visible. An empty or developing item must
never look like a finished course note.
