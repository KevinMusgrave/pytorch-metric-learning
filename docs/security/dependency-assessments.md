# Development dependency assessments

Assessor: Codex. Reviewed: 2026-10-26. Scope: the exact versions in `uv.lock`.
These assessments do not authorize publication or establish CUDA acceptance.

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
expiring 2026-11-25, except certifi's critical-severity assessment, which the
policy limits to seven days per review: it was re-verified on 2026-10-28 and
expires on 2026-11-04. `mkdocs.yml` still enables no plugins and no repository
source imports Requests. Repeat that check weekly until 2026-11-19, when the
fixed release reaches 30 days and should replace the aged one. Reassess before
adding the affected APIs or deployment modes.

| Dependency        | Advisory                                 | Enabled path and absent prerequisite                                                                                                                                                                                                                                                                                                         |
| ----------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| filelock 4.0.3    | GHSA-v4x8-3wr5-3wq8, GHSA-9vhv-jxww-xfm5 | PyTorch's `torch.utils._filelock.FileLock` wraps `filelock.FileLock`. On supported Mac/Linux hosts it selects `UnixFileLock`; Windows selects `WindowsFileLock`. The SQLite-backed `ReadWriteLock` APIs are not used.                                                                                                                        |
| filelock 4.0.3    | GHSA-567q-5qvw-pf92                      | The affected Windows `SoftFileLock` path is not selected by those callers. No repository source calls it directly.                                                                                                                                                                                                                           |
| setuptools 84.0.0 | GHSA-xmj3-9gf7-h52w                      | The build uses `setuptools.build_meta`; uv installs dependency wheels. There is no `setup_requires`, egg installation, or call to `Wheel.install_as_egg`, which consumes the malicious namespace metadata.                                                                                                                                   |
| certifi 2026.7.22 | GHSA-6q48-6c7w-9gw2                      | Requests enters the graph through MkDocs Material. Inspection of its 9.7.7 wheel finds Requests calls only in the info, privacy, and social plugins; none is enabled in `mkdocs.yml`. The library makes no network requests, and package acquisition uses the separately patched uv. No enabled caller uses certifi's affected trust bundle. |

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

## Filelock local filesystem advisories

Assessor: Codex. Reviewed: 2026-10-28. These additional assessments cover
`filelock==4.0.3` in the frozen development environment and expire on 2026-11-25.
The advisories were published on 2026-10-28; the resolved dependency did not
change. Their prerequisites were checked against the installed filelock and
PyTorch 2.14.0 sources, repository callers, CI configuration, and local cache
permissions.

The library, tests, and recipes do not call filelock directly. Installed
callers are PyTorch's compiler cache and debug code, which select `FileLock`,
not the SQLite, strict-soft, or soft-reader/writer classes. The reviewed
macOS and Linux filesystems support native `flock`; the `ENOSYS` soft fallback
is not selected. Ordinary tests and CI benchmarks use eager execution.
Compiled benchmarks on the reviewed Mac use the default Inductor cache below
the user's mode-0700 temporary directory. Other local accounts cannot replace
its lock paths. CI uses an isolated GitHub-hosted Linux runner; GPU acceptance
uses a disposable Linux pod with no untrusted local accounts.

| Advisory            | Affected API or prerequisite absent from this environment                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| GHSA-fj95-fr6c-h845 | `SoftReadWriteLock` holder-record refresh; this API is not called.                                                                     |
| GHSA-j8f7-rjxc-mr56 | SQLite `ReadWriteLock` symlink resolution; this API is not called.                                                                     |
| GHSA-m462-7rxw-wvpg | `StrictSoftFileLock` protocol-directory traversal; this API is not called.                                                             |
| GHSA-9j6q-wvj5-jgjg | Soft-lock writability checks under attacker-controlled paths; no soft-lock caller or fallback is enabled.                              |
| GHSA-69hj-f68j-2879 | Native Unix locks can truncate a file planted at the lock path. The reviewed cache directories are not writable by untrusted accounts. |
| GHSA-8mx5-7m5m-r77g | Windows lock ACL inheritance; no Windows execution or SQLite lock is enabled in this development baseline.                             |
| GHSA-6627-m3xw-93mp | Writable strict-soft claim directories; this API is not called.                                                                        |
| GHSA-fq56-3cx3-fjxp | Soft-lock stale-record inode reuse; no soft-lock caller or fallback is enabled.                                                        |
| GHSA-9c4x-gm8v-wgc4 | Adoption of attacker-owned strict-soft or soft-reader/writer protocol directories; neither API is called.                              |
| GHSA-rfpc-qcw5-3pc4 | Forged soft-lock holder records; no soft-lock caller or fallback is enabled.                                                           |
| GHSA-cwqr-hh33-gh44 | Crafted SQLite lock databases; this API is not called.                                                                                 |

The exact advisory URLs are retained with each assessment in
`.code-polishy.json`. Additional source references are PyTorch's
[cache implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/runtime/caching/implementations.py)
and [cache paths](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/runtime/cache_dir_utils.py),
and filelock's [native Unix implementation](https://github.com/tox-dev/filelock/blob/4.0.3/src/filelock/_unix.py).

These findings are not a claim that filelock 4.0.3 is safe in arbitrary consumer
applications. Reassess before enabling Windows verification, soft-lock APIs,
compiler caches writable by untrusted users, shared-user GPU hosts, or a
filesystem without native `flock`. Keep compilation caches private on such
hosts. Filelock 4.2.1 fixes this set and was published on 2026-10-17; review
that release or a later aged fix from 2026-11-16. No early-release admission is
needed while the affected inputs and APIs remain absent.

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

## Removed dataset and evaluation dependencies

Removing the dataset adapters, testers, and model inference wrapper removed
Pillow, torchvision, and tqdm from `uv.lock`. The Pillow 13.0.0 release-age
admission existed only because dataset image decoding reached its native
codecs; it was removed with the dependency. SciPy is no longer a direct
dependency: the large-margin softmax coefficients use exact integer binomials.
scikit-learn, and through it SciPy, remains only in the optional `clustering`
extra and the development group for NMI/AMI tests.

## Documentation math renderer

The documentation site loads MathJax 3.2.2 (Apache-2.0), published to npm on
2022-06-08, from
`https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js`. The npm
tarball matches its registry integrity
`sha512-Bt+SSVU8eBG27zChVewOicYs7Xsdt40qm4+UpHyX7k0/O9NliPc+x77k1/FEsPsjKPZGJvtRZM1vO+geW0OhGw==`,
and the CDN file is byte-identical to that tarball's `es5/tex-mml-chtml.js`.
`overrides/main.html` pins the script with SHA-384 subresource integrity, so a
changed CDN file is refused by the browser.

MathJax 4 was not selected: it loads font data as additional scripts that the
page cannot pin with integrity hashes. Version 3 uses its default
configuration, which typesets the `arithmatex` output of `pymdownx.arithmatex`.
Its TeX `autoload` extension would fetch further scripts from the same
version-pinned CDN path, without integrity hashes, only for macros outside the
bundled packages. The rendered-page check records which network requests the
documented equations make.

MathJax runs in readers' browsers over documentation authored in this
repository; no build or test step executes it. The uv lock and Code Polishy
supply-chain checks do not cover this browser asset, so update it by repeating
this verification.
