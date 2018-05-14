# TUL
Identifying Human Mobility via Trajectory Embeddings
#Implement
1. word2vec:
We use the pip tool 'pip install word2vec', I have remove the POIs which the frequency is less than a threshold. So I have uploaded this file in github (See XX-scopus.dat). So the 'min-count' must be 0, otherwise, you will miss some POIs.
 From my understanding of pure semi-supervised learning (never know which are test data) which we have proposed in our paper, I tried to embedding all of the POIs including all of the labelled data and unlabelled data (just like scopus). (Important: remove the user ID from gowalla_scopus_1104.dat)
command as: 'word2vec -train gowalla_scopus_real.dat output gowalla_em_250.dat -size 250 -window 5 -min-count 0 -cbow 0'
2. Training process:
We choose the 201 users' sub-trajectories, split these to  training data(about 90) and test data (about 10%).
# Related Literature

<br>
Qiang Gao,Fan Zhou,Kunpeng Zhang,Goce Trajcevski,Xucheng Luo,Fengli Zhang.Identifying Human Mobility via Trajectory Embeddings.The 26th International Joint Conference on Artificial Intelligence (IJCAI'17).
