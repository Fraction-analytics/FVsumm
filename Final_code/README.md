# Deep Reinforcement Learning for Unsupervised Video Summarization with Determinantal point process.
# Implement with python = 3.8

## Requirement

python = 3.8

Pytorch

tabulate

## 1. Get Started
```bash
mkdir dataset and put mp4 video files 
```
## 2. Dataset Generation
```bash
python create_data.py --input videos --output dataset/data.h5
``` 
## 3. Feature Extraction
```bash
python test.py -d dataset/data.h5
``` 
## 4. Summary Generation
Single video

```bash
python test.py -d dataset/data.h5
```
## 4.1 Summary Generation
For Multi videos

```bash
python test_final.py -d dataset/video1.h5 -da dataset/video2.h5 -f data/video1 -fa data/video2
```
Video summarization will be saved folder log/summary
