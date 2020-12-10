from .batch_easy_hard_miner import BatchEasyHardMiner


class BatchHardMiner(BatchEasyHardMiner):
    def __init__(self, **kwargs):
        super().__init__(
            pos_strategy=BatchEasyHardMiner.HARD,
            neg_strategy=BatchEasyHardMiner.HARD,
            **kwargs
        )
