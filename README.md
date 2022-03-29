# PPI_GNN
In order to replicate the results mentioned in paper, please follow the following steps:
  1. Download the Pan's human features file and place the files at ../human_features/processed/. The link is given in PPI_GNN/Human_features/README.md. For the S.      cerevisiae PPI dataset, download the input feature file and place it at ../S. cerevisiae/processed/. The link is given in PPI_GNN/S. cerevisiae/README.md.
  2. Next use the command: python train.py to train the model.


The steps to predicting protein interactions on a new dataset are:
  1. First, get the node features from protein sequences using the SeqVec method (seqvec_embedding.py) and then build the protein graph (proteins_to_graphs.py).
  2. Next, use the command "python data_prepare.py" to get input features for the model.
  3. Then, use the command "python train.py" to train the model.
  4. Use the command: "python test.py" to evaluate the trained model on unseen data (test set).
