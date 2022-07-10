# VSE: Virtue State Embedding for Knowledge Graph Completion

## Usage

To train and evaluate a KG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {FB15K,WN,umls,kinship}]
              [--model {InceptE}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug]

Knowledge Graph Embedding

optional arguments:
  -h, --help            show this help message and exit
  --dataset {B15K,WN,umls,kinship}
                        Knowledge Graph dataset
  --model {InceptE}
                        Knowledge Graph embedding model
  --regularizer {N3,N2}
                        Regularizer
  --reg REG             Regularization weight
  --optimizer {Adagrad,Adam,SparseAdam}
                        Optimizer
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --rank RANK           Embedding dimension
  --batch_size BATCH_SIZE
                        Batch size
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias {constant,learn,none}
                        Bias type (none for no bias)
  --dtype {single,double}
                        Machine precision
  --double_neg          Whether to negative sample both head and tail entities
  --debug               Only use 1000 examples for debugging
```

## Acknowledgement
We refer to the code of ATTH. Thanks for their great contributions!

## New models

```
def get_queries(self, queries):
    head_e = self.entity(queries[:, 0])
    rel_e = self.rel(queries[:, 1])
    lhs_e = ### Do something here ###
    lhs_biases = self.bh(queries[:, 0])
    return lhs_e, lhs_biases
```
