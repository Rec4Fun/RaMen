# RaMen
This is implementation of the paper RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction, accepted at ECAI 2025

# How to train model
```
python train.py --dataset {dataset_name}
```

# Hyperparams Setting
In the final version of paper at ECAI2025, we provide full details of hyperparameter tuning to facilitate reproducibility. 

* -g: GPU number
* -wandb: use wandb to log result or not
* -bundle_alpha: hyperparameter in bundle-level contrastive loss 
* -item_alpha: hyperparameter in item-level contrastive loss
* -hyper_num: number of hypergraph edges
* -alpha_residual: hyperparameter of residual item in asymmetric module 
* -n_layer_gat: number of layers in enhanced item module 
