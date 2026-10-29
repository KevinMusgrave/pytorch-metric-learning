# Versioning and support

Releases follow [semantic versioning](https://semver.org/) for the public API.
Raising a minimum supported Python, PyTorch, or NumPy version is not treated as
a breaking change; it happens only in minor or major releases:

- **Major** versions may remove or rename public APIs and change default
  numerical behavior. There are no compatibility aliases or deprecation
  shims; each major release has release notes and a migration guide.
- **Minor** versions add features and may raise the minimum supported Python,
  PyTorch, or NumPy version. Changed minimums are listed in the release notes.
- **Patch** versions fix bugs without changing public signatures. A fix that
  changes a loss value is described in the release notes.

The public API is everything documented on this site: the `distances`,
`losses`, `miners`, `reducers`, `regularizers`, `samplers`, `evaluation`,
`distributed`, `types`, and `utils.indexing` modules. Names starting with an
underscore and other `utils` modules are private and can change in any release.

## Supported environments

| Component      | Supported                                                            |
| -------------- | -------------------------------------------------------------------- |
| Python         | 3.12, 3.13, 3.14                                                     |
| PyTorch        | 2.14.0 and newer                                                     |
| NumPy          | 2.5.3 and newer                                                      |
| CPU platforms  | Linux, macOS, and Windows; each is tested in CI                      |
| CUDA           | Linux with PyTorch's CUDA wheels; tested on two GPUs before releases |
| Faiss backends | `faiss-cpu` 1.15.1 and newer; GPU search needs a GPU Faiss build     |

The minimum versions are the ones CI tests against. Only the latest release
receives fixes. The final version 2 release is 2.9.0, documented in the
[version 2 documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/v2/).
