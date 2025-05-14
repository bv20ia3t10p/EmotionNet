# Test Dataset Directory

This directory contains a minimal structure for testing the emotion recognition model. For actual training, you need to populate it with real image data.

## Directory Structure

```
test_dataset/
├── fer2013/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── val/
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprise/
└── raf-db/
    ├── train/
    │   ├── 1_angry/
    │   ├── 2_disgust/
    │   ├── 3_fear/
    │   ├── 4_happy/
    │   ├── 5_neutral/
    │   ├── 6_sad/
    │   └── 7_surprise/
    └── val/
        ├── 1_angry/
        ├── 2_disgust/
        ├── 3_fear/
        ├── 4_happy/
        ├── 5_neutral/
        ├── 6_sad/
        └── 7_surprise/
```

## How to Add Real Data

1. **For FER2013 Dataset**:
   - Download the FER2013 dataset from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
   - Extract the images and organize them according to the directory structure above

2. **For RAF-DB Dataset**:
   - Request access to the RAF-DB dataset from: http://www.whdeng.cn/raf/model1.html
   - Organize the aligned faces according to the directory structure above

## Alternative: Using FER2013 CSV File

If you have the FER2013 dataset as a CSV file:

1. Place the `fer2013.csv` file directly in the `test_dataset/fer2013/` directory
2. The training script will automatically detect and process the CSV file 