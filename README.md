# CL-GNN
Project for the "Machine Learning Algorithms and Applications" course at TU Wien 2023W.\
*Advisors: Tamara Drucks, Pascal Welke*, TU-Wien.

### General Idea
The aim of this project is to identify whether distance preserving embeddings for graphs can be obtained. To investigate this, a Siamese Network architecture is used, and Graph Neural Networks (GNNs) are used to handle graph data. This approach could be exploited as a self-supervised step to perform on a GNN before using it for some other downstream task.\
To obtain a distance between graphs these are represented through their vectors of expectation-complete homomorphism counts, as proposed in Welke et al. (2023) [1].

The base layer consists of $k$ Graph Convolutional layers (GCN), as proposed by Kipf et al. (2017) [2], followed by $m$ linear layers. Additionally, non-linear and batch normalisation layers are included, while global mean pooling is used to move from a node-level to a graph-level representation. The network base layer is shown in the image below, in this case when working with two input graphs:

![alt text](https://github.com/SDavenia/CL-GNN/blob/main/images/architecture.png)


To train the network, standard contrastive learning ideas are exploited and adapted to try and preserve distance. To be more specific, the network can be trained using either one of two losses:
- Contrastive loss idea (CL): the network takes as input a pair of graphs which are fed through the same layers to obtain graph embeddings. Afterwards, the distance between those is computed, and the network is penalised if this distance is different from the one computed on the vector of homomorphism counts or densities of the two graphs.
- Triplet loss idea (TL): the network takes as input a triple of graphs $(a, p, n)$, where $a$ is the anchor, $p$ is a positive example (i.e. should be similar to the anchor) and $n$ is a negative example (i.e. should be different from the anchor). To try and preserve the distance the following adjustement to a standard Triplet Loss is made. Denote as $d_p$ the distance between the anchor and positive graph embeddings and $d_n$ that between the anchor and the negative graph embedding. The network is penalised when the difference $d_p - d_n$ is smaller in the embedding space (in absolute terms) than when computed on the initial vectors of homomorphism counts.

### Usage
To run the code call `main.py` as shown below. Ensure that the file containing the desired homomorphism counts is present in `data/homcounts` and is named `<dataset>_<nhoms>.homson`:
```
python main.py --loss <loss> --dataset <dataset> --nhoms <number_of_homomorphisms> --n_conv_layers <num_GCN_layers> --n_lin_layers <num_Linear_layers>  --distance <distance> --hom_types <hom_types>
```

where:
- `loss`: is one of \[`contrastive`, `triplet`\] specifies whether training should be performed using the CL or TL approach.
- `dataset`: dataset to be used. For the experiments only `MUTAG` and `ENZYMES` were considered.
- `number_of_homomorphisms`: specifies the number of patterns used to obtain homomorphism counts.
- `num_GCN_layers`: specifies the number of GCN layers to use in the architecture.
- `num_Linear_layers`: specifies the number of Linear layers to use in the architecture.
- `distance`: is one of \[`L1`, `L2`, `cosine`\] and specifies the distance to use both for the vectors of homomorphism counts and the embeddings.
- `hom_types`: is one of \[`counts`, `counts_density`\] and specifies whether vectors of homomorphism counts or homomorphism densities should be used.

> Note that additional parameters to control training are also available (learning rate, batch size, ...). Check file `main.py` for further information.

### Structure of the repository
- `Utilities.py`: contains utility functions for plotting and preparing dataloaders.
- `models.py`: contains the implementation of the architecture described above.
- `training.py`: contains the functions used for training and validation.
- `exploration_MUTAG.ipynb`, `exploration_ENZYMES.ipynb`: contain an exploratory analysis of the distances obtained using homomorphism counts.
- `data`: contains the json format files storing homomorphism counts in `data/homomorphism_counts` and patterns files which were analyzed.
- `results`: contains the plots of actual vs predicted distances in `results/actual_vs_predicted` and the losses obtained during the training procedure in `results/train_val_loss`.


### References
[1] Welke, P., \& Thiessen, M., \& Jogl, F., \& Gärtner, T. (2023). Expectation-complete graph representations with homomorphisms. In *International Conference on Machine Learning*.\
[2] Kipf, T.N., \& Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In *Proceedings of the 5th International Conference on Learning Representations*.
