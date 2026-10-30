# Security policy

## Supported versions

Security fixes are released for the latest 3.x version.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/KevinMusgrave/pytorch-metric-learning/security/advisories/new).
Do not open a public issue. Include the affected version, a minimal
reproduction, and the impact you observed.

The library loads no untrusted files on its own. Checkpoints that contain its
modules, such as `CrossBatchMemory` state, should be loaded with
`torch.load(..., weights_only=True)`.
