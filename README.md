# Microstructure Classification - CNN vs. Logistic Regression

A deep learning project for classifying steel microstructures from microscopy images using a Convolutional Neural Network (CNN) built in PyTorch.

The notebook is written in an educational style - each step includes detailed comments explaining what is happening and why, making it useful for anyone learning deep learning or PyTorch from scratch.

---

## Dataset

The project uses the [UHCS (Ultra-High Carbon Steel) Microstructure dataset](https://www.kaggle.com/datasets/sagarupsc/uhcs-microstructure-01), available on Kaggle.

The dataset contains 961 microscopy images of steel microstructures with labels stored in a SQLite database. Four pure microstructure classes were used:

| Class | Images |
|---|---|
| spheroidite | 374 |
| network | 212 |
| pearlite | 124 |
| martensite | 36 |

Mixed classes (e.g. pearlite+spheroidite) were excluded to keep the classification problem clean and interpretable. The dataset is imbalanced - martensite has significantly fewer samples, which affects model performance on that class.

---

## Project Pipeline

1. **Data exploration** - dataset structure, SQLite database, class distribution
2. **Preprocessing** - resize to 128x128, normalization, data augmentation (flips, rotation)
3. **Custom PyTorch Dataset and DataLoader** - 80/20 train/test split with stratification
4. **CNN architecture** - 3 convolutional blocks (16/32/64 filters) + fully connected layers + dropout
5. **Training** - 40 epochs, Adam optimizer, CrossEntropyLoss, best model checkpointing
6. **Evaluation** - confusion matrix, classification report (precision, recall, F1)
7. **Learning curves** - accuracy and loss over epochs
8. **Comparison** - CNN vs. Logistic Regression baseline

---

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 51.3% |
| **CNN** | **84.7%** |

Logistic Regression treats each pixel as an independent feature and cannot capture spatial patterns. CNN detects local structures (edges, textures, microstructural features) through convolutional layers - hence the 33.4 percentage point difference.

Per-class CNN performance:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| spheroidite | 0.88 | 0.88 | 0.88 |
| network | 0.88 | 0.88 | 0.88 |
| pearlite | 0.69 | 0.80 | 0.74 |
| martensite | 1.00 | 0.43 | 0.60 |

Martensite scores lowest on recall due to limited training data (29 images) - the model is precise when it predicts martensite, but misses most of the actual martensite samples.

---

## Technologies

- **Python 3**
- **PyTorch** - CNN architecture, training loop
- **torchvision** - image transforms, augmentation
- **scikit-learn** - Logistic Regression baseline, metrics
- **Pandas / NumPy** - data manipulation
- **Matplotlib / Seaborn** - visualizations
- **SQLite** - label storage and retrieval
- **Pillow** - image loading

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sagarupsc/uhcs-microstructure-01) and unzip it.

2. Place the following files in the same folder:
```
your_folder/
├── microstructure_classification.ipynb
├── microstructures.sqlite
├── best_model.pth
└── micrographs/         <- folder with images from the dataset
```

3. Open the folder in Jupyter Notebook and run `microstructure_classification.ipynb` cell by cell.

> Note: the notebook must be run from the folder containing `microstructures.sqlite` and the `micrographs` folder.

---

## Possible Improvements

- Hyperparameter tuning (learning rate, batch size, number of filters)
- Larger dataset or data synthesis for underrepresented classes (martensite)
- Class weighting to address class imbalance
