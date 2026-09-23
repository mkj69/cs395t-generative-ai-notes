# Notebook architecture

## The central decision

Use two note systems without mixing their purposes.

- **Learning notes** follow lecture order and record manual reconstruction.
- **Research notes** follow questions and preserve original inquiry.

Do not split Markdown, Python, figures, and references into unrelated format
silos inside either kind of note.

The learning unit is a lecture reconstruction:

```text
course-notes/lecture-XX-short-topic/
├── note.md              # recall, derivation, correction, and uncertainty
├── code/                # checks written by the author
└── figures/             # explanatory figures created by the author
```

The research unit is a question. Its prose and supporting artifacts stay close
together:

```text
notes/<question-slug>/
├── note.md              # the argument and current conclusion
├── code/                # scripts or notebooks used only by this note
├── figures/             # explanatory and generated figures
└── data/README.md       # provenance; large/raw data stays external
```

## Five public surfaces

1. **Learning notes** follow the course while separating recall, source facts,
   derivation, correction, and unresolved questions.
2. **Research threads** connect several notes and experiments around a larger
   question. A thread is a map, not another essay.
3. **Research notes** develop one question through intuition, derivation,
   evidence, connections, and open problems.
4. **Experiments** make a claim reproducible. They record code, configuration,
   environment, expected output, and interpretation—including negative
   results.
5. **Source library** records papers and other materials actually used. It
   stores citations, claim ledgers, and reading trails, but not downloaded PDFs.

## Repository map

```text
cs395t-generative-ai-notes/
├── course-notes/          # lecture-ordered manual reconstruction
│   └── _template/
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

- Put a lecture-specific check in `course-notes/<lecture>/code/` and record the
  expected result before running it.
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
lecture source
    ↓
manual capture + closed-source reconstruction
    ↓
checked learning note
    ↓
question worth pursuing
    ↓
seed research note + local experiment
    ↓
evidence and citation check
    ↓
reviewed research note
    ↓
public index + relevant research thread
```

The website should make these states visible. A `capturing` or `reconstructing`
learning note must never look checked, and an empty research template must
never look like a finished course note.
