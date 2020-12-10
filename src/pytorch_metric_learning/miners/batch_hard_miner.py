from .batch_easy_hard_miner import BatchEasyHardMiner


class BatchHardMiner(BatchEasyHardMiner):
    def __init__(self, **kwargs):
        super().__init__(
            positive_strategy=BatchEasyHardMiner.HARD,
            negative_strategy=BatchEasyHardMiner.HARD,
            **kwargs
        )
