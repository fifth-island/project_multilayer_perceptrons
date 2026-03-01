# Fashion MNIST Garment Classification with MLPs

A deep learning project exploring multi-layer perceptron (MLP) architectures for 10-class garment classification on the Fashion MNIST dataset. The project investigates the role of activation functions (GELU vs ReLU), regularization strategies (dropout, batch normalization, L2), and iterative model refinement through three progressive architectures.

---

## Project Structure

```
classifying_fashion_MNIST_MLP/
├── mlp_project.ipynb               # Full experiment notebook
├── HW3-1_report.md                 # Blog-post style writeup
├── HW3-1_paper.tex                 # LaTeX academic paper source
├── paper_mlp.pdf                   # Compiled PDF paper
├── figures/                        # All exported static figures
│   ├── 00_sample_grid.png
│   ├── 01_class_distribution.png
│   ├── 02_mean_images.png
│   ├── 03a_pca_2d.png
│   ├── 03b_umap_2d.png
│   ├── 04_gelu_vs_relu.png
│   ├── 05_before_preprocessing.png
│   ├── 06_after_preprocessing.png
│   ├── 07_training_curves_model1.png
│   ├── 08_training_curves_model2.png
│   ├── 09_confusion_matrix_model1.png
│   ├── 10_training_curves_model3.png
│   ├── 11_confusion_matrix_final.png
│   ├── 12_misclassified_examples.png
│   ├── 13_all_models_comparison.png
│   ├── 14_per_class_f1.png
│   └── 15_umap_before_after.png
└── README.md
```

---

## Dataset

**Fashion MNIST** — 70,000 grayscale images (28×28) across 10 balanced clothing categories.

| Split | Samples |
|-------|---------|
| Train | 60,000 |
| Test  | 10,000 |

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

---

## Models

Three MLP architectures trained progressively, each building on lessons from the previous:

| Model | Architecture | Key Techniques | Val Accuracy |
|-------|-------------|----------------|-------------|
| Model 1 | 784 → 256 → 128 → 10 | ReLU, dropout | ~85% |
| Model 2 | 784 → 512 → 256 → 128 → 10 | GELU, dropout, L2 | ~88% |
| Model 3 (Final) | 784 → 512 → 256 → 128 → 64 → 10 | GELU, batch norm, dropout, L2 | ~89% |

---

## Key Findings

- **GELU consistently outperforms ReLU** on this dataset — smoother gradients reduce training oscillation and improve generalization
- **Portfolio regularization** (dropout + batch normalization + L2 together) is more effective than any single technique alone
- **Hardest classes**: Shirt, T-shirt/top, Pullover, and Coat — visually similar upper-body garments with overlapping pixel distributions
- **UMAP reveals structure**: learned embeddings show tight, well-separated clusters for easy classes (Trouser, Bag) and overlapping clusters for confusable classes
- **Performance ceiling ~89–90%** for MLPs on this dataset; CNNs typically reach 93–95% by preserving spatial structure

---

## Outputs

| File | Description |
|------|-------------|
| `mlp_project.ipynb` | Interactive notebook with Plotly visualizations + all experiments |
| `HW3-1_report.md` | Blog-style writeup with embedded figures |
| `HW3-1_paper.tex` | Full academic paper (LaTeX source) |
| `paper_mlp.pdf` | Compiled PDF of the academic paper |

---

## Running the Notebook

The notebook is designed for **Google Colab** but runs locally with the same environment.

**Dependencies** (installed automatically in the first cell):

```bash
pip install umap-learn plotly tensorflow
```

**To regenerate figures**, run all cells top to bottom. Figures are saved to `figures/` automatically.

**To recompile the LaTeX paper** (after regenerating figures):

```bash
pdflatex HW3-1_paper.tex
pdflatex HW3-1_paper.tex   # second run resolves cross-references and ToC
```

---

## Tech Stack

- **Python** 3.12
- **TensorFlow / Keras** 3.x
- **UMAP-learn** — dimensionality reduction for learned representations
- **Plotly** — interactive charts (notebook)
- **Matplotlib / Seaborn** — static figures (report/paper)
- **scikit-learn** — PCA, classification metrics
