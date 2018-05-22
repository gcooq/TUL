# TUL
Identifying Human Mobility via Trajectory Embeddings
# Environment
* python 2.7
* Tensorflow 1.0 or ++ （updated now 2018.05.22)
# Usage
* word2vec:
Use the pip tool 'pip install word2vec'，The data tha we have removed the POIs which the frequency is less than a threshold.
For pure semi-supervised learning (never know which are test data) which we have proposed in our paper,  embedding all of the POIs including all of the labelled data and unlabelled data (just like scopus). (Important: remove the user ID from gowalla_scopus_1104.dat)
command as: 'word2vec -train gowalla_scopus_real.dat output gowalla_em_250.dat -size 250 -window 5 -min-count 0 -cbow 0'
* Training process:
We choose the 201 users' sub-trajectories, split these to  training data(about 90) and test data (about 10%).
The new code with tensorflow>=1.0, you can run it easily. and also some records will stored by the code (including model, train data and sample results), you can download from：

https://drive.google.com/file/d/128fCjfKPcqnKFhuYZiRFIqKmNnXx5NsZ/view?usp=sharing

For GRU_S, GRU, LSTM_S, you just change the fucntionin 'RNN', tensorflow has GRU, 2-layer GRU.
# Related Literature

<br>
Qiang Gao,Fan Zhou,Kunpeng Zhang,Goce Trajcevski,Xucheng Luo,Fengli Zhang.Identifying Human Mobility via Trajectory Embeddings.The 26th International Joint Conference on Artificial Intelligence (IJCAI'17).
