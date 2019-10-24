# pytorch_metric_learning

Installation:
```
pip install pytorch_metric_learning
```

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

The package also comes with RecordKeeper, which makes it very easy to log and save data during training. It automatically looks for special attributes within objects to log on Tensorboard, as well as to save in CSV and pickle format.
```
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.utils import record_keeper as record_keeper_package

pickler_and_csver = record_keeper_package.PicklerAndCSVer(your_folder_for_logs)
tensorboard_writer = SummaryWriter(log_dir=your_tensorboard_folder)
record_keeper = record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver)

# Then during training:
recorder.update_records(your_dict_of_objects, current_iteration)

# If you are using one of the provided trainers, then just pass in the record keeper, and the update step will be taken care of.
trainer = trainers.MetricLossOnly(
  <your other args>
  record_keeper = record_keeper
  ...
)

# Now it will update the record_keeper at every iteration
trainer.train()
```

The nice thing about RecordKeeper is that it makes it very easy to add loggable information when you write a new loss function or miner. Just create a list named "record_these" that contains the names of the attributes you want to record.
```
class YourNewLossFunction(BaseMetricLossFunction):
  def __init__(self, **kwargs):
    self.avg_embedding_norm = 0
    self.some_other_useful_stat = 0
    self.record_these = ["avg_embedding_norm", "some_other_useful_stat"]
    super().__init__(**kwargs)
    
  def compute_loss(self, embeddings, labels, indices_tuple):
    self.avg_embedding_norm = torch.mean(torch.norm(embeddings, p=2, dim=1))
    self.some_other_useful_stat = some_cool_function(embeddings)
```

