# pytorch_metric_learning

## See this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1kiJ5rKmneQvnYKpVO9vBFdMDNx-yLcXV2wbDXlb-SB8/edit?usp=sharing) for benchmark results (in progress). 
## See [powerful_benchmarker](https://github.com/KevinMusgrave/powerful_benchmarker/) to use the benchmarking tool.

## Loss functions implemented:
- angular
- contrastive
- lifted structure
- margin
- multi similarity
- n pairs
- nca
- proxy nca
- triplet margin
- **more to be added**

## Mining functions implemented:
- distance weighted sampling
- hard aware cascaded mining
- maximum loss miner
- multi similarity miner
- pair margin miner
- **more to be added**

## Training methods implemented:
- metric loss only
- training with classifier
- cascaded embeddings
- deep adversarial metric learning
- **more to be added**

## Installation:
```
pip install pytorch_metric_learning
```


## Overview

Use a loss function by itself
```python
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(normalize_embeddings=False, margin=0.1)
loss = loss_func(embeddings, labels)
```

Or combine miners and loss functions, regardless of whether they mine or compute loss using pairs or triplets. Pairs are converted to triplets when necessary, and vice versa.
```python
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(normalize_embeddings=False, margin=0.1)
hard_pairs = miner(embeddings, labels)
loss = loss_func(embeddings, labels, hard_pairs)
```

Train using more advanced approaches, like deep adversarial metric learning. For example:
```python
from pytorch_metric_learning import trainers

# Set up your models, optimizers, loss functions etc.
models = {"trunk": your_trunk_model, 
          "embedder": your_embedder_model,
          "generator": your_negative_generator}

optimizers = {"trunk_optimizer": your_trunk_optimizer, 
              "embedder_optimizer": your_embedder_optimizer,
              "generator_optimizer": your_negative_generator_optimizer}
              
loss_funcs = {"metric_loss": losses.AngularNPairs(alpha=35),
              "synth_loss": losses.Angular(alpha=35), 
              "g_adv_loss": losses.Angular(alpha=35)}

mining_funcs = {}

loss_weights = {"metric_loss": 1, 
                "classifier_loss": 0,
                "synth_loss": 0.1,
                "g_adv_loss": 0.1,
                "g_hard_loss": 0.1,
                "g_reg_loss": 0.1}

# Create trainer object
trainer = trainers.DeepAdversarialMetricLearning(
  models=models,
  optimizers=optimizers,
  batch_size=120,
  loss_funcs=loss_funcs,
  mining_funcs=mining_funcs,
  num_epochs=50,
  iterations_per_epoch=100,
  dataset=your_dataset,
  loss_weights=loss_weights
)
  
trainer.train()
```
