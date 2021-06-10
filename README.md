# Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward.
Implement with python=3.x https://github.com/KaiyangZhou/pytorch-vsumm-reinforce

<div align="center">
  <img src="img/pipeline.jpg" alt="train" width="80%">
</div>

## Requirement
python=3.x

Pytorch

GPU

tabulate

## Get started
```bash
mkdir dataset
```

1. Requirements to Generate Dataset

Put your videos in folder dataset
```bash
Create a folder name dataset and put videos in this folder
```

2. Dataset Generation
```bash
python create_data.py --input dataset --output dataset/data.h5
```

3. Find Determinant of I and J
```bash
python test.py -d dataset/data.h5
```

4. Diversity and Representation score calculation
```bash
python test.py -d dataset/data.h5
```

5. Reducing redundant frames 
```bash
python test.py -d dataset/data.h5
```

Video summarization will be saved folder log/final_summary
