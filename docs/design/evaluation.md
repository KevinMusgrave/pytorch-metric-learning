# Retrieval and evaluation

Testers extract embeddings and delegate nearest-neighbor metrics to the accuracy
calculator. Query and reference sets remain explicit: self-matches are excluded
only when the query occurs in the reference set. Label comparisons may be
customized and need not be scalar equality.

Faiss is an optional retrieval backend. The reproducible development environment
uses `faiss-cpu`, including when embeddings originate on CUDA. CUDA tensor tests
and two-device collective tests therefore do not require an obsolete `faiss-gpu`
wheel. Backend-specific GPU acceleration is distinct from PyTorch CUDA support.

Logging hooks write local records and TensorBoard event files. They do not start
an HTTP server. Training consumes these hooks; evaluation must not import the
trainer implementations. Metric definitions and query/reference examples live
in `docs/accuracy_calculation.md` and `docs/testers.md`.
