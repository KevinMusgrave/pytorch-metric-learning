# Frequently Asked Questions

## Large batch sizes and the `INT_MAX` error

A large batch size results in a huge number of pairs/triplets. When the `loss_and_miner_utils` code processes a huge number of tuples, it can cause a PyTorch error:

`RuntimeError: nonzero is not supported for tensors with more than INT_MAX element`

To fix this error, lower your batch size.
