# Development dependency assessments

Assessor: Codex. Reviewed: 2026-10-26. Scope: the exact versions in `uv.lock`.
These assessments do not authorize publication or establish CUDA acceptance.

## Pillow 13.0.0

`BaseDataset.__getitem__` calls `PIL.Image.open` on paths derived from dataset
metadata and then invokes a supplied transform. Dataset directories and image
contents can come from outside the repository. The initial 12.3.0 resolution
therefore exposed native decoders, rather than merely containing an unused
Pillow dependency. In particular,
[GHSA-78g4-xqwv-q3g8](https://github.com/python-pillow/Pillow/security/advisories/GHSA-78g4-xqwv-q3g8)
reaches a bundled OpenJPEG heap write through ordinary open/load calls, even
when a file has a JPEG or PNG extension. Other affected decoding paths include
GIF, PNG, TIFF, and JPEG processing. Public reproductions exist; this review
does not claim exploitation of this repository.

The official [13.0.0 release](https://github.com/python-pillow/Pillow/releases/tag/13.0.0)
is the first release fixing the observed advisory set. Its wheels were published
on 2026-10-16. An older Pillow avoids some newly introduced defects but remains
affected by others whose ranges begin at version zero. Restricting filename
extensions does not restrict codecs, and a JPEG/PNG format restriction does not
address every affected transform or decoder. Disabling image loading would
remove the dataset adapter's purpose. No aged release provides the required
fixes across these paths.

Admit 13.0.0 through 2026-11-15, when it reaches 30 days. Waiting would retain
publicly described native memory-corruption paths for approximately 20 days.
The admitted release has additional nonsecurity changes, including removals and
bundled native-library updates; its age and binary provenance remain residual
supply-chain risks. Use official PyPI wheels with the lock's hashes, disable
source builds during dependency installation, and verify the dataset/image
behavior with local fixtures. No source fork or locally rebuilt Pillow is used.

## Certificate trust

[uv GHSA-hq85-5cpj-mr62](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62)
describes compromised intermediate CA keys under the emSign ECC Root CA - G3.
An on-path attacker with one of those keys can impersonate HTTPS origins.
uv uses its bundled roots for package acquisition, an enabled security-sensitive
path. The separate certifi advisory is assessed below against the actual
documentation plugin configuration.

Admit `uv==0.13.4` through 2026-11-19.
This exact release was uploaded on 2026-10-20 and removes the affected trust.
There is no aged fixed release. Disabling TLS verification is unacceptable.
Switching every client to an independently maintained custom trust bundle would
add a second security maintenance responsibility and would require proving the
affected root absent on every development and CI host.

The uv
[fix](https://github.com/astral-sh/uv/pull/22760) updates
`webpki-root-certs` from 1.0.9 to 1.0.10. Its release also changes preview wheel
variant validation; that preview is not enabled here. The broader uv release
contains resolver and credential-handling changes, so a clean security scan
alone does not establish its safety. Exact artifact hashes, frozen resolution,
disabled dependency builds, and normal lock/install verification limit the
admission. Continuing to trust a CA with compromised keys while downloading
executable dependencies is the greater risk during the remaining 24 days.

## Retained aged dependencies

The following advisories remain visible under exact `not-affected` assessments
expiring 2026-11-25, except certifi's critical-severity assessment, which must be
reviewed again by 2026-11-02. Reassess before adding the affected APIs or deployment modes.

| Dependency        | Advisory                                 | Enabled path and absent prerequisite                                                                                                                                                                                                                                                                                                |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| filelock 4.0.3    | GHSA-v4x8-3wr5-3wq8, GHSA-9vhv-jxww-xfm5 | PyTorch's `torch.utils._filelock.FileLock` wraps `filelock.FileLock`. On supported Mac/Linux hosts it selects `UnixFileLock`; Windows selects `WindowsFileLock`. The SQLite-backed `ReadWriteLock` APIs are not used.                                                                                                               |
| filelock 4.0.3    | GHSA-567q-5qvw-pf92                      | The affected Windows `SoftFileLock` path is not selected by those callers. No repository source calls it directly.                                                                                                                                                                                                                  |
| setuptools 84.0.0 | GHSA-xmj3-9gf7-h52w                      | The build uses `setuptools.build_meta`; uv installs dependency wheels. There is no `setup_requires`, egg installation, or call to `Wheel.install_as_egg`, which consumes the malicious namespace metadata.                                                                                                                          |
| certifi 2026.7.22 | GHSA-6q48-6c7w-9gw2                      | Requests enters the graph through MkDocs Material. Inspection of its 9.7.7 wheel finds Requests calls only in the info, privacy, and social plugins; none is enabled in `mkdocs.yml`. Library downloads use urllib, and package acquisition uses the separately patched uv. No enabled caller uses certifi's affected trust bundle. |

The source evidence includes PyTorch
[FileLock](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/utils/_filelock.py),
[Inductor callers](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/codecache.py),
filelock's [platform selection](https://github.com/tox-dev/filelock/blob/4.0.3/src/filelock/__init__.py),
and setuptools' [legacy wheel installer](https://github.com/pypa/setuptools/blob/v84.0.0/setuptools/wheel.py).

For certifi, the official 2026.10.19 wheel fixes the trust problem and reaches
30 days on 2026-11-19. Its verified comparison with 2026.7.22 changes only
`cacert.pem` and the version declaration, removing the advisory's emSign ECC
Root CA - G3. Because the affected client is not enabled here, retain the aged
release until then instead of admitting a fresh package unnecessarily. Enabling
any of the named MkDocs plugins or adding a Requests caller invalidates this
assessment and requires review before use.

## CUDA metadata coverage

PyTorch's Linux dependency resolves `cuda-toolkit==13.0.3.0`. PyPI's version
endpoint accepts that equivalent version but reports `info.version` as
`13.0.3`, with wheel `cuda_toolkit-13.0.3.0-py2.py3-none-any.whl` published on
2026-04-14. See the
[authoritative metadata](https://pypi.org/pypi/cuda-toolkit/13.0.3.0/json).
Code Polishy 0.31.3 reported a missing exact release timestamp for the lock's
spelling. Code Polishy 0.31.4 matches the PEP 440 version identity, and the
online supply-chain check passed on 2026-10-27 UTC with the existing lock and
assessments. No exception is needed for this release. The dependency is
Linux-only and is not installed in the Mac CPU environment; metadata acceptance
does not establish CUDA runtime verification.

## Removed logging dependencies

Removing the trainer/logging integration removed record-keeper, TensorBoard,
absl-py, grpcio, protobuf, tensorboard-data-server, and Werkzeug from `uv.lock`.
The exact Werkzeug advisory assessment was removed with the dependency. This
reduces the installed surface rather than adding an advisory exemption.
