# pytorch_metric_learning

## Coming soon in a separate repo: a highly configurable and easy-to-use benchmarking tool.

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
```
from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(normalize_embeddings=False, margin=0.1)
loss = loss_func(embeddings, labels)
```

Or combine miners and loss functions, regardless of whether they mine or compute loss using pairs or triplets. Pairs are converted to triplets when necessary, and vice versa.
```
from pytorch_metric_learning import miners, losses
miner = miners.MultiSimilarityMiner(epsilon=0.1)
loss_func = losses.TripletMarginLoss(normalize_embeddings=False, margin=0.1)
hard_pairs = miner(embeddings, labels)
loss = loss_func(embeddings, labels, hard_pairs)
```

Train using more advanced approaches, like deep adversarial metric learning. For example:
```
from pytorch_metric_learning import trainers

# Set up your models, optimizers, loss functions etc.
models = {"trunk": your_trunk_model, 
          "embedder": your_embedder_model,
          "G_neg_model": your_negative_generator}

optimizers = {"trunk_optimizer": your_trunk_optimizer, 
              "embedder_optimizer": your_embedder_optimizer,
              "G_neg_model_optimizer": your_negative_generator_optimizer}
              
loss_funcs = {"metric_loss": losses.AngularNPairs(alpha=35),
              "synth_loss": losses.Angular(alpha=35), 
              "G_neg_adv": losses.Angular(alpha=35)}

mining_funcs = {}

loss_weights = {"metric_loss": 1, 
                "classifier_loss": 0,
                "synth_loss": 0.1,
                "G_neg_adv": 0.1,
                "G_neg_hard": 0.1,
                "G_neg_reg": 0.1}

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
  
# Train!
trainer.train()
```
