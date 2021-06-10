# Deep Reinforcement Learning for Unsupervised Video Summarization with Determinantal point process.
# Implement with python=3.x 

## Requirement

python=3.x

Pytorch

tabulate

## 1. Get started
```bash
mkdir dataset and put mp4 video files 
```
## 2. Dataset Generation
```bash
python create_data.py --input dataset --output dataset/data.h5
```

## 3. Find Determinant 
```bash
python test.py -d dataset/data.h5
```

## 4. Diversity and Representation calculation
```bash
python test.py -d dataset/data.h5
```

## 5. Reducing redundant frames 
```bash
python test.py -d dataset/data.h5
```

Video summarization will be saved folder log / final_summary
