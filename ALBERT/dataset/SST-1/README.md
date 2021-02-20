# [SST-1](https://nlp.stanford.edu/sentiment/)



### 原数据集说明

> Stanford Sentiment Treebank V1.0
>
> This is the dataset of the paper:
>
> Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
> Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
> Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
>
> If you use this dataset in your research, please cite the above paper.
>
> @incollection{SocherEtAl2013:RNTN,
> title = {{Parsing With Compositional Vector Grammars}},
> author = {Richard Socher and Alex Perelygin and Jean Wu and Jason Chuang and Christopher Manning and Andrew Ng and Christopher Potts},
> booktitle = {{EMNLP}},
> year = {2013}
> }
>
> This file includes:
> 1. original_rt_snippets.txt contains 10,605 processed snippets from the original pool of Rotten Tomatoes HTML files. Please note that some snippet may contain multiple sentences.
>
> 2. dictionary.txt contains all phrases and their IDs, separated by a vertical line |
>
> 3. sentiment_labels.txt contains all phrase ids and the corresponding sentiment labels, separated by a vertical line.
> Note that you can recover the 5 classes by mapping the positivity probability using the following cut-offs:
> [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
> for very negative, negative, neutral, positive, very positive, respectively.
> Please note that phrase ids and sentence ids are not the same.
>
> 4. SOStr.txt and STree.txt encode the structure of the parse trees. 
> STree encodes the trees in a parent pointer format. Each line corresponds to each sentence in the datasetSentences.txt file. The Matlab code of this paper will show you how to read this format if you are not familiar with it.
>
> 5. datasetSentences.txt contains the sentence index, followed by the sentence string separated by a tab. These are the sentences of the train/dev/test sets.
>
> 6. datasetSplit.txt contains the sentence index (corresponding to the index in datasetSentences.txt file) followed by the set label separated by a comma:
> 	1 = train
> 	2 = test
> 	3 = dev
>
> Please note that the datasetSentences.txt file has more sentences/lines than the original_rt_snippet.txt. 
> Each row in the latter represents a snippet as shown on RT, whereas the former is each sub sentence as determined by the Stanford parser.
>
> For comparing research and training models, please use the provided train/dev/test splits.



### 子目录

| 文件夹                                                       | 数据               |
| ------------------------------------------------------------ | ------------------ |
| [stanfordSentimentTreebank](./stanfordSentimentTreebank)     | **标注的数据集**   |
| [stanfordSentimentTreebankRaw](./stanfordSentimentTreebankRaw) | **未标注的数据集** |
| [trainDevTestTrees_PTB](./trainDevTestTrees_PTB)             |                    |



### 文件

| 文件                               | 详情         |
| ---------------------------------- | ------------ |
| [data_process.py](data_process.py) | 数据处理代码 |
| [train.tsv](train.tsv)             | **训练集**   |
| [test.tsv](test.tsv)               | **测试集**   |
| [dev.tsv](dev.tsv)                 | **验证集**   |



### 数据集说明

| 名称               | 说明                      |
| ------------------ | ------------------------- |
| **sentence_index** | 句子的编号（没啥用）      |
| **sentence**       | 句子                      |
| **label**          | 句子标签【0，1，2，3，4】 |

* **最大句长**：56

* **平均句长**：18.89

* **句子长度分布**：

  ![sentence](\.md\sentence_length_distr.png)

* **单词频率分布**：

  ![words](\.md\word_show_distr.png)