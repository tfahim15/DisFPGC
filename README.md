# Discriminating Frequent Pattern based Supervised Graph Embedding for Classification

This repository is the official implementation of the paper titled "Discriminating Frequent Pattern based Supervised Graph Embedding for Classification"

## Datasets
<b>Table:</b> Description of datasets

| Dataset Name | No. of graphs  |Avg no. of nodes|Avg no. of edges|No. of classes|No. of node labels|No. of edge labels
|---------|---|---|---|---|---|---|
|DD|1,178|284.32|715.66|2|82|-|
|ENZYMES|600|32.63|62.14|6|3|-|
|IMDB-B|1,000|19.77|96.53|2|-|-|
|MUTAG|188|17.93|19.79|2|7|11|
|NCI1|4,110|29.87|32.30|2|37|3|
|NCI109|4,127|29.68|32.13|2|38|3|
|PROTEINS|1,113|39.06|72.82|2|3|-|
|PTC|344|14.29|14.69|2|19|-|

All the datasets are available in the <b>data/&lt;dataset_name&gt;/</b> folder. In each folder, there are two files: <b>&lt;dataset_name&gt;_graph.txt</b>(e.g. dd_graph.txt) and  <b>&lt;dataset_name&gt;_label.txt</b>(e.g. dd_label.txt).

<b>&lt;dataset_name&gt;_graph.txt</b>:
```
t # <graph_id>
v <vertex_id> <vertex_label>
...
e <vertex_id> <vertex_id> <edge_label>
...
```
<b>&lt;dataset_name&gt;_label.txt</b>:
```
<graph_label> <graph_id>
...
```
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Codes

All the codes are available in the <b>code/&lt;dataset_name&gt;/</b> folder. In each folder, there are three files:
```codes
mine_subgraphs.py
train_eval.py
cover_freq.txt
```
<b>mine_subgraphs.py:</b> contains codes for mining candidate feature subgraphs. Mining can be skipped as the output of mining is saved in <i>cover_freq.txt</i>. To run,
```mine_subgraphs
> cd code/<dataset_name>
> python mine_subgraphs.py
```
<b>train_eval.py:</b> contains codes for training embedding, classifier and then evaluate by predicting test set for 10 folds. It takes candidate feature subgraphs as input from file <i>cover_freq.txt</i> and outputs average accuracy. To run,
```train_eval
> cd code/<dataset_name>
> python train_eval.py
```

<b>cover_freq.txt:</b> contains pre mined candidate feature subgraphs and graphs covered by them. The structure is as follows:
```  
 # <DFS code of candidate subgraph>
 <graph_id> <count_of_subgraph_isomorphisms>
...
```


## Results

Our model achieves the following performance:

| Dataset Name | Accuracy|
|---------|---|
|DD| 93.46| 
|ENZYMES|57.42|
|IMDB-B|88.56|
|MUTAG| 97.00|
|NCI1|94.92|
|NCI109|97.40|
|PROTEINS|83.44|
|PTC|84.94|
