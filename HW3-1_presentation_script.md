# HW3.1 Code Review Presentation Script
> **Note:** Cell numbers match the notebook order. All cells are covered, including EDA, preprocessing visualizations, architecture diagrams, GELU explainer, comparison dashboard, and learned representation visualizations.

---

## Introduction & Goal

- **The challenge:** build an MLP (Multi-Layer Perceptron) to classify Fashion MNIST images and achieve **>90% test accuracy**.
- Fashion MNIST was created by Zalando Research in 2017 as a drop-in replacement for the classic MNIST handwritten digits — the original MNIST was considered "too easy" (most models hit 97%+ trivially), so Zalando created a harder version with the exact same format.
- 10 clothing classes, 60,000 training images + 10,000 test images, each 28×28 grayscale pixels.
- **Fun fact:** The dataset is intentionally limited to 28×28 pixels — fine details like buttons, zippers, and collar styles are largely invisible. This makes certain classes (Shirt vs. T-shirt vs. Pullover) genuinely ambiguous even to the human eye.

---

## Cell 1 — Install Libraries

```python
!pip install -q plotly umap-learn
```

- **What it does:** Installs two libraries not available by default in Colab:
  - **Plotly:** Interactive charting library — produces zoomable, hoverable, animated charts that render inline in notebooks. Far more presentation-friendly than static matplotlib.
  - **UMAP (Uniform Manifold Approximation and Projection):** A nonlinear dimensionality reduction algorithm. We'll use it later to project 784-dimensional image data into 3D space to visualize how images cluster by class.
- The `-q` flag suppresses verbose output to keep the notebook clean.

---

## Cells 2–3 — Introduction & Dataset Description

- Sets up the problem context: this is a continuation of HW2.1 (basic MNIST MLPs), now applied to the harder Fashion MNIST.
- Introduces the 10 class labels (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) with numeric codes 0–9.
- **Point to make:** The numeric labels are purely arbitrary — there is no mathematical ordering between the classes. Class 9 (Ankle Boot) is not "greater than" Class 0 (T-shirt). This is why we use one-hot encoding later instead of raw integers.

---

## Cell 4 — Loading the Data

```python
from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```

- **What it does:** Loads the dataset directly from Keras's built-in dataset registry. On first run it downloads and caches it locally (~30 MB).
- `X_train` contains 60,000 images, `X_test` contains 10,000 images.
- `y_train` and `y_test` contain the corresponding integer class labels (0–9).
- **Why this way:** Keras wraps several benchmark datasets behind a one-liner API, ensuring reproducibility — everyone who runs this gets the exact same split.
- **Trivia:** The same dataset is available via `torchvision.datasets.FashionMNIST` in PyTorch, via TensorFlow Datasets as `fashion_mnist`, and via Hugging Face Datasets. The format is fully compatible with original MNIST code.

---

## Cell 6 — Sample Image Grid

```python
img = X_train[y_train == i][0]
plt.imshow(img, cmap='gray', interpolation='none')
```

- **What it shows:** A 2×5 grid displaying one example image per class, rendered in grayscale.
- **Why `cmap='gray'`:** Each pixel is a single intensity value (no color channels). `gray` maps 0 → black (background), 255 → white (foreground clothing).
- **Why `interpolation='none'`:** Disables anti-aliasing so you see the actual raw pixels without smoothing — important for building intuition about the data's true resolution.
- **Key observation to make:** Visually compare Shirt (6), T-shirt/top (0), Pullover (2), and Coat (4) — even to a human eye, these are hard to tell apart at 28×28. Meanwhile, Trouser, Bag, and Sandal are visually distinct — and the model will classify those nearly perfectly.

---

## Cell 8 — EDA: Class Distribution Bar Chart

```python
fig = px.bar(df_dist, x="Class", y="Count", color="Class", title="Training Set Class Distribution (60,000 images)")
```

- **What it shows:** An interactive bar chart with each of the 10 classes on the x-axis and their sample count on the y-axis.
- **What the values represent:** Each bar height = the number of training images in that class. The text labels on top of each bar show the exact count.
- **What to point out:** Every single class has exactly **6,000 samples** — a **perfectly balanced dataset**. This is intentional (Zalando curated it this way).
- **Why balance matters:** In an **imbalanced dataset**, a model can achieve high accuracy by simply predicting the majority class every time. For example, if 90% of images were T-shirts, a model saying "it's always a T-shirt" would get 90% accuracy while being completely useless. Because Fashion MNIST is perfectly balanced, accuracy is a meaningful metric — the model genuinely has to correctly identify all 10 classes.
- **(hover over bars for exact counts; click to zoom)**

---

## Cell 9 — EDA: Mean Images + Pixel Intensity Distributions

**Part 1: Mean Image Grid (matplotlib, 2×5 heatmap)**

```python
mean_img = X_train[y_train == i].mean(axis=0)
ax.imshow(mean_img, cmap='hot', interpolation='bilinear')
```

- **What it shows:** For each class, we average all 6,000 training images pixel-by-pixel to get the "prototype" — the most representative image of that class.
- **What the colors represent:** The `hot` colormap maps low pixel values → black/red, high values → bright yellow/white. This makes average intensity contrast more visible than plain gray.
- **What to point out:** The mean images for Shirt, T-shirt, Pullover, and Coat look almost identical — blurry blobs of brightness centered on the upper body. This isn't a data issue; it reflects *genuine visual similarity* at this resolution. Contrast with Trouser (distinct vertical shape) and Sandal (distinct low-profile shape).
- **Key insight:** If the average images of two classes look nearly the same, no MLP can reliably separate them based only on pixel values. This sets a performance ceiling that we'll see confirmed by the model.

**Part 2: Pixel Intensity Distribution (interactive overlaid histograms)**

```python
fig.add_trace(go.Histogram(x=pixels_sample, name=class_names[i], opacity=0.6, nbinsx=50))
```

- **What it shows:** For each class, a histogram where:
  - **X-axis:** pixel intensity value (0 = black/background, 255 = white/bright foreground)
  - **Y-axis:** how many pixels have that intensity (frequency)
  - All 10 classes overlaid on the same chart
- **What the distribution shape tells us:**
  - A spike at **0** means most pixels in that class are black (background)
  - A second bump at higher values (100–255) represents the actual clothing pixels
  - The **bimodal shape** (two humps) is characteristic of Fashion MNIST — sparse foreground object on a clean black background
  - Trouser tends to have more white pixels (the whole shape is filled); Sandal has less filled-area
- **(click legend to toggle individual classes; hover for exact counts)**

---

## Cell 11 — 3D PCA Scatter

```python
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_sample)
fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Class', ...)
```

- **What it shows:** 10,000 randomly sampled training images (1,000 per class) projected from 784 dimensions down to 3 using **PCA**, displayed as a 3D scatter. Each dot = one image; color = class.
- **What PCA (Principal Component Analysis) is:** PCA finds the directions in the data with the most variation and projects the data onto those directions. Think of it as finding the most informative "shadows" of the 784-dimensional cloud. PC1 captures the most variance, PC2 the second most, etc.
  - The axis values (PC1, PC2, PC3) are not interpretable as pixel values — they are projections onto abstract mathematical axes. What matters is *relative distance between points*.
- **What "Explained variance by 3 components: ~XX%" means:** The percentage of the original 784-dimensional information preserved by these 3 components. Typically 30–40% for Fashion MNIST — meaning we're discarding 60–70% of the original information for visualization purposes.
- **What to look for:**
  - Tight single-color clusters = images of that class are visually similar and distinct from others → easy to classify
  - Colors blending together = those classes overlap in pixel space → hard to classify
  - **Expected pattern:** Trouser forms a clear elongated cluster; Bag and Ankle Boot are relatively distinct; Shirt/T-shirt/Pullover/Coat form an overlapping central blob.
- **(rotate by clicking and dragging to explore the structure from all angles)**

---

## Cell 12 — 3D UMAP Scatter

```python
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap_3d = reducer.fit_transform(X_pca_50)
```

- **What it shows:** The same 10,000 images projected to 3D using **UMAP** instead of PCA.
- **What UMAP is and how it differs from PCA:**
  - PCA is **linear** — finds straight-line projections and preserves global variance structure, producing smeared overlapping clouds.
  - UMAP is **nonlinear** — builds a graph of nearest neighbors and preserves *local* neighborhood structure. Points similar in high-dimensional space stay close in 3D; dissimilar points are pushed apart.
  - **Pipeline:** 784D → PCA to 50D (fast, removes noise dimensions) → UMAP to 3D (nonlinear quality compression).
- **Parameter meanings:**
  - `n_neighbors=15`: Each point considers its 15 nearest neighbors when building the local graph. Smaller → finer local clusters; larger → broader global structure.
  - `min_dist=0.1`: Controls how tightly points cluster in the embedding. Smaller → denser clusters.
- **What to look for:** UMAP produces much tighter, more separated clusters than PCA. You should see clear "islands" for Trouser, Sneaker, Sandal, Ankle Boot, Bag. Upper-body clothing classes still form a merged region — this is **genuine ambiguity**, not a UMAP limitation.
- **Key message:** Even with the best possible 3D visualization, some classes are inseparable. Any model operating on 784-pixel grayscale images shares this limitation.

---

## Cell 13 — Animated Rotating 3D UMAP

```python
frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.5*np.cos(...))))) for angle in range(0,360,5)]
```

- **What it shows:** The same UMAP 3D scatter from the previous cell, animated to rotate a full 360° around the vertical axis. Press ▶ Play to start; ⏸ Pause to stop.
- **Why this is useful:** A single static 3D view can be misleading — clusters that appear separated from one angle might overlap from another. The rotation reveals the full 3D structure and prevents misreading the embedding.
- **Technical note:** The animation is 72 camera positions (one every 5°) stored as Plotly `Frame` objects, played back via `updatemenus` Play/Pause buttons. No video file needed — runs entirely in the browser.
- **(press Play for the presentation — the rotating view shows cluster structure far more convincingly than any static screenshot)**

---

## Cell 14 — 3D UMAP with Image Thumbnails on Hover

```python
hover_images.append(f'<img src="data:image/png;base64,{b64}" width="84">')
```

- **What it shows:** A subset of 200 UMAP embedding points (20 per class) where hovering over any dot shows the *actual clothing image* at that position.
- **How it works:** Each 28×28 image is encoded as a base64 PNG string (a self-contained HTML-compatible image format) and stored in the Plotly hover text. Plotly renders the `<img>` tag inline when you hover.
- **Why this closes the loop:** It connects abstract math (UMAP coordinates) to concrete reality (the actual garment). You can hover over neighboring dots and directly see *why* those garments are considered similar — often revealing visually ambiguous items sitting at the boundary between clusters.
- **What to demo live:** Hover over points at the edge of the Shirt cluster. You'll see garments that genuinely could be classified as either Shirt or T-shirt — making future model errors understandable in advance.

---

## Cell 17 — Data Inspection & Save Raw Copies

```python
X_train_raw = X_train.copy()  # Save before preprocessing overwrites X_train
print("X_train shape:", X_train.shape)   # (60000, 28, 28)
print("Pixel range:", X_train.min(), "-", X_train.max())  # 0 - 255
```

- **What it does:**
  1. Saves raw copies of the data *before* preprocessing transforms them in Cell 19
  2. Prints shape and pixel range statistics as a sanity check
- **Output:** `X_train` is shape `(60000, 28, 28)` — 60,000 images, each a 28×28 2D array. Pixel values are integers ranging 0–255.
- **Why save raw copies:** Cell 19 preprocesses `X_train` in-place (reshape, normalize, one-hot encode), permanently changing its format. We save `X_train_raw` now so before/after visualizations have access to the original 28×28 integer images — and so later UMAP cells can reference unprocessed images for hover thumbnails.

---

## Cell 18 — BEFORE Preprocessing Visualization

*3-panel figure: raw image heatmap | zoomed crop with pixel values | pixel histogram*

**Panel 1 — Raw image heatmap:**
```python
ax.imshow(sample_img, cmap='gray')
plt.colorbar(im, ax=ax)
```
- **What it shows:** The first training image as a grayscale heatmap. The colorbar on the right maps pixel intensity to the gray scale: 0 = pure black (background), 255 = pure white (brightest foreground).
- **What to point out:** Each pixel is just a number. The image is visually clear but very low resolution — fine details like fabric texture or collar style are completely lost.

**Panel 2 — Zoomed 8×8 crop with pixel values annotated:**
```python
ax.text(j, i, str(crop[i, j]), color='red' if crop[i,j] < 128 else 'yellow')
```
- **What it shows:** A magnified 8×8 section of the image with the actual integer pixel value printed on each cell. Dark pixels (< 128) show values in red; bright pixels (≥ 128) show in yellow.
- **What to point out:** Raw data is literally a grid of integers. There's nothing special here — each pixel is just a number between 0 and 255. This makes concrete *why* we normalize: feeding "200" vs "0.78" to a neural network makes a huge difference to gradient magnitudes.

**Panel 3 — Pixel intensity histogram:**
```python
ax.hist(sample_img.flatten(), bins=50)
```
- **What it shows:** X-axis = pixel value (0–255); Y-axis = how many pixels have that value.
- **The bimodal shape** (two humps) reveals: a large spike at 0 (most pixels = black background) and a secondary cluster at higher values (clothing object). This bimodal structure is characteristic of Fashion MNIST's clean black background design.
- **Why it matters for training:** Unnormalized inputs with values like 200–255 create disproportionately large gradient updates compared to 0-valued background pixels, destabilizing training. Normalization (Cell 19) corrects this.

---

## Cell 19 — Preprocessing

```python
X_train = X_train.reshape(X_train.shape[0], 784)   # Flatten
X_train = X_train / 255.0                           # Normalize to [0, 1]
y_train = to_categorical(y_train, 10)               # One-hot encode
```

**Three critical transformations:**

1. **Flatten (28×28 → 784 — shape changes from (60000,28,28) to (60000,784)):**
   - Converts each 2D image matrix into a 1D vector of 784 values. MLPs require flat vector input — they have no concept of spatial position or pixel neighborhoods.
   - **This is a fundamental limitation of MLPs on images:** a pixel at the edge is treated identically to a pixel at the center. CNNs (Convolutional Neural Networks) preserve spatial relationships and outperform MLPs significantly on image data for this reason.

2. **Normalize (÷ 255 — values change from [0,255] integers to [0.0,1.0] floats):**
   - During backpropagation, gradients are proportional to input values. Large raw pixel values (0–255) create large, uneven gradient updates that make training unstable and slow.
   - Keeping all inputs in [0,1] ensures balanced gradient flow across all 784 features and typically leads to faster, more stable convergence.

3. **One-hot encoding (integer 3 → vector [0,0,0,1,0,0,0,0,0,0]):**
   - `categorical_crossentropy` loss compares the model's 10-probability output against the target. The target must also be a 10-element probability distribution (which one-hot labels are: one 1.0 and nine 0.0s).
   - Without one-hot encoding, the model would incorrectly treat labels as ordinal numbers, implying "Ankle Boot (9) is 9× more important than T-shirt (0)" — which is nonsensical for unordered categories.

---

## Cell 20 — AFTER Preprocessing Visualization

*3-panel figure: normalized image | flattened vector waveform | one-hot bar chart*

**Panel 1 — Normalized image reshaped to 28×28:**
- **What it shows:** The same image from Cell 18, now with float32 values in [0.0, 1.0]. The colorbar confirms the new range.
- **What to point out:** The image looks *identical* — normalization doesn't change visual content, only the numeric scale. This is key: it's a purely mathematical transformation that doesn't alter what the data represents.

**Panel 2 — Flattened 784D vector (first 200 values plotted):**
```python
ax.plot(processed_img[:200], color='steelblue')
ax.fill_between(range(200), processed_img[:200], alpha=0.3)
```
- **What it shows:** X-axis = pixel index (0–199 of 784); Y-axis = normalized pixel intensity (0.0–1.0). The shaded area beneath the line fills the profile for readability.
- **What the waveform represents:** This is the literal input vector the neural network receives. Spikes represent bright clothing pixels; valleys near zero represent background. The network sees 784 independent float values — no concept of the 2D arrangement.
- **Why show this:** Makes the abstract "784-dimensional input" concrete. The model processes this entire waveform as one vector with no spatial context.

**Panel 3 — One-hot encoded label:**
```python
bars = ax.bar(range(10), processed_label, color=['red' if v==1 else 'blue' for v in processed_label])
```
- **What it shows:** 10 bars (one per class); all are zero (blue) except the true class bar which is 1.0 (red).
- **What this represents:** The target vector the model trains toward. During training, the loss function measures how far the model's 10 predicted probabilities diverge from this ideal "100% confident correct class" distribution.

---

## Cell 22 — Interactive Architecture Diagram Helper (`draw_mlp_diagram`)

```python
def draw_mlp_diagram(model, title=None):
    # Plotly-based interactive neuron-and-connections diagram
```

- **What it produces:** An interactive diagram showing:
  - **Gray column (leftmost):** Input layer — number above = 784 (one per pixel)
  - **Blue columns (middle):** Hidden Dense layers — number above = neuron count, activation shown below
  - **Green column (rightmost):** Output layer — 10 neurons, one per class, softmax activation
  - **Lines between columns:** Synaptic connections representing weights. Each visible neuron connects to every neuron in the next layer — this is what "fully connected" (Dense) means
  - **Hover tooltip:** Shows exact layer name and unit count
- **Design choices:**
  - Caps at 8 neurons shown per column with `···(N total)` note — showing all 512 neurons would make the diagram unreadable
  - Dropout layers are intentionally omitted — they're a training-time regularizer, not an architectural component (at inference time, they're completely inactive)
- **What "fully connected" means structurally:** For a 784→512 connection, every one of the 784 input neurons sends a signal to every one of the 512 neurons = 784 × 512 = **401,408 weight parameters** just for that one connection. This is why parameter counts are in the hundreds of thousands.

---

## Cell 23 — Model 1: Baseline MLP (535,818 parameters)

```python
model1 = Sequential([
    keras.Input(shape=(784,)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
], name="Model1_Baseline")
```

**Architecture (as shown in the interactive diagram below the summary):**
Input (784) → Dense-512 (relu) → Dense-256 (relu) → Dense-10 (softmax)

**Design choices explained:**

- **ReLU activation (Rectified Linear Unit):** Formula: `f(x) = max(0, x)`. Passes positive values through unchanged; converts any negative value to 0.
  - **Why use it:** Computationally cheap, avoids the **vanishing gradient problem** (gradients becoming exponentially smaller in deep networks), and was the key activation innovation enabling deep learning around 2012.
  - **Limitation — Dead Neurons:** If a neuron receives a consistently large negative input, its ReLU output and gradient are both exactly 0. That neuron stops learning permanently — "dies." This motivates the GELU switch in Model 3.

- **Softmax output:** Converts 10 raw output scores into a probability distribution summing to 1.0. Each output = the model's estimated probability for that class. The highest probability is the prediction.

- **Dropout(0.3):** During training, randomly zeroes out 30% of neurons in the preceding layer each forward pass.
  - **Why it prevents overfitting:** **Overfitting** = the model memorizes training examples instead of learning generalizable patterns. Dropout forces redundant pathways — no single neuron can carry critical information alone — producing a more generalizable model.
     -- Cool that this is inspired in the biomimetics of how brain neurons actual work.
  - At inference/test time, Dropout is disabled; all neurons are active.

- **Adam optimizer:** Maintains per-parameter adaptive learning rates based on first and second gradient moment estimates. Consistently outperforms plain SGD without needing manual learning rate tuning.

- **categorical_crossentropy loss:** Standard loss for multi-class classification with one-hot labels. Measures the KL-divergence between the model's predicted probability distribution and the true one-hot distribution.

- **Training:** 20 epochs, batch_size=128, 20% validation split.

---

## Cell 24 — Model 1 Training

```python
hist1 = model1.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
```

- **Epochs:** One epoch = one full pass through all 60,000 training images (in batches of 128 = 469 batches). 20 epochs = 20 full passes.
- **Validation split:** 20% of training data (12,000 images) is held out — never used for weight updates — only for measuring performance after each epoch. This gives an honest estimate of how the model generalizes to unseen data.
- **`hist1`:** The returned history object stores `categorical_accuracy`, `val_categorical_accuracy`, `loss`, and `val_loss` per epoch — used in the next cell.

---

## Cell 25 — Model 1 Training Curves

*2-panel interactive Plotly chart: Accuracy (left) | Loss (right)*

```python
fig.add_trace(go.Scatter(y=hist1.history['categorical_accuracy'], name='Train Acc'))
fig.add_trace(go.Scatter(y=hist1.history['val_categorical_accuracy'], name='Val Acc', line=dict(dash='dash')))
```

**What each panel shows:**
- **Left — Accuracy:** Y-axis = proportion of correctly classified images (0.0–1.0), X-axis = epoch. Solid = training accuracy, dashed = validation accuracy.
- **Right — Loss:** Y-axis = categorical crossentropy loss value. Lower is better. Both should decrease over training.

**Key concept — Overfitting:**
- **Overfitting** = the model memorizes training data rather than learning generalizable patterns.
- Visual signs: train accuracy keeps rising while **validation accuracy plateaus or falls**; train loss falls while **validation loss stops decreasing or rises**.
- The **overfitting gap** printed below = `train_accuracy − val_accuracy`. A gap > 0.05 is a warning sign.
- **For Model 1:** The 30% Dropout limits memorization effectively, producing a moderate gap. However, this aggressive Dropout may also prevent the model from fully using its capacity — potentially causing it to plateau before reaching 90%.

---

## Cell 27 — Model 1 Evaluation

**Test accuracy: ~88.6%**

**Classification report — metric definitions:**
- **Precision:** Of all images the model labeled as class X, what fraction actually were class X? ("Are the model's predictions trustworthy for class X?")
- **Recall:** Of all images that truly belong to class X, what fraction did the model catch? ("How thorough is the model at finding class X?")
- **F1-score:** Harmonic mean of precision and recall: `2 × (P × R) / (P + R)`. A balanced single metric where a perfect score = 1.0. The harmonic mean penalizes lopsided precision/recall more than the arithmetic mean would.
- **Support:** Number of actual test images per class (always 1,000 for Fashion MNIST).

**Confusion matrix — how to read it:**
- A 10×10 grid. Row i = true class; column j = predicted class.
- The **diagonal** = correct predictions → these cells should be darkest.
- **Off-diagonal cells** = misclassifications. The brighter the cell, the more errors of that type.
- **What to highlight:** Row 6 (Shirt) has conspicuous off-diagonal brightness — Shirts are frequently predicted as T-shirts, Pullovers, and Coats. This directly confirms the ambiguity we saw in the EDA mean images.

---

## Cell 29 — Model 2: Larger ReLU MLP (798,474 parameters)

```python
model2 = Sequential([
    keras.Input(shape=(784,)),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax'),
], name="Model2_Larger")
```

**Changes from Model 1 and rationale:**

| Change | Model 1 | Model 2 | Hypothesis |
|--------|---------|---------|------------|
| Hidden layers | 2 | 3 | More layers = more capacity to represent complex patterns |
| Dropout rate | 0.3 each | 0.1 once | Model 1's dropout was too aggressive; try less regularization |
| Dropout positions | After each hidden layer | Only once | Less regularization overall |
| Epochs | 20 | 60 | More complex model needs more time |
| Val split | 0.2 | 0.1 | Give model more training examples |

- **Result: ~89–90%** — marginally better than Model 1 but doesn't reliably cross the target.
- **What the training curves reveal:** With only 0.1 Dropout and no other regularization, Model 2 starts overfitting after ~20–30 epochs. The validation loss curve diverging upward while training loss continues falling is the clear signal.
- **Lesson:** Simply increasing capacity without smarter regularization mainly increases memorization, not generalization.

---

## Cell 31 — Model 2 Training Curves

- **What to look for vs. Model 1:** By epoch 20–30, validation accuracy typically plateaus or reverses while training accuracy continues rising — the **overfitting signature**.
- The val loss curve rising while train loss falls is the clearest visual evidence of overfitting.
- **Contrast with Model 1:** Model 1's curves track more closely (smaller gap) because stronger Dropout limited memorization — at the cost of slightly lower peak accuracy.
- **Key message:** Adding capacity (more layers, less Dropout) bought small accuracy gains at the cost of overfitting risk. We need smarter regularization techniques — not just fewer neurons randomly dropped.

---

## Cell 33 — GELU vs. ReLU Explainer Chart

*2-panel interactive Plotly chart: Activation Functions (left) | Derivatives/Gradients (right)*

```python
gelu = x * norm.cdf(x)          # GELU: x times standard normal CDF
relu = np.maximum(0, x)          # ReLU: max(0, x)
gelu_deriv = norm.cdf(x) + x * norm.pdf(x)
relu_deriv = np.where(x > 0, 1.0, 0.0)
```

**Left panel — Activation Functions (what goes in vs. what comes out):**
- **X-axis:** input value to the activation function (ranges −4 to +4)
- **Y-axis:** output value
- **ReLU (blue):** perfectly linear for x > 0 (slope = 1), hard zero for all x ≤ 0. The sharp corner at x = 0 is mathematically non-differentiable.
- **GELU (red):** smooth curve, slightly negative for small negative x (values like −0.1 to −0.17), curving up to approximately linear for large positive x. No discontinuity.
- **Swish/SiLU (green dotted):** similar idea (sigmoid gating), included for context — used in EfficientNet and MobileNetV3.
- The dashed gray diagonal (identity: y = x) shows where each function diverges from "pass everything through unchanged."

**Right panel — Derivatives (gradient flow during backpropagation):**
- **X-axis:** input value; **Y-axis:** the gradient (derivative) of the activation at that input
- **Why gradients matter:** During backpropagation, the activation function's gradient is multiplied into the error signal flowing backward through the network. If gradient = 0, no learning signal propagates through that neuron.
- **ReLU gradient (blue):** A step function — exactly **0 for all x < 0**, exactly **1 for all x > 0**.
  - **Dead neuron problem:** A neuron that repeatedly receives negative inputs gets gradient = 0 every time. Its weights never update. It is effectively "dead" and contributes nothing — wasted model capacity.
- **GELU gradient (red):** Always **positive and nonzero** for any finite input. Even strongly negative inputs have a tiny but non-zero gradient. Neurons always have some learning signal and can never permanently die.
- **Why this matters for Fashion MNIST:** The ambiguous classes (Shirt vs. T-shirt) require fine-grained decision boundaries. GELU's smooth gradient landscape helps the optimizer make precise weight adjustments, whereas ReLU's hard cutoffs create a rougher landscape with potential dead zones near those critical boundaries.
- **Trivia:** GELU (Gaussian Error Linear Unit) was proposed by Hendrycks & Gimpel (2016). The formula `x · Φ(x)` (where Φ is the standard normal CDF) gates how much input to pass through based on whether it's "likely positive" under a Gaussian prior. It is now the standard activation in GPT, BERT, and all modern transformer architectures.

---

## Cell 35 — Model 3: Final GELU Model (912,610 parameters) ✅

```python
model3 = Sequential([
    keras.Input(shape=(784,)),
    Dense(600, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=l2(1e-5)),
    Dropout(0.06),
    Dense(512, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=l2(1e-5)),
    Dropout(0.06),
    Dense(256, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=l2(1e-5)),
    Dropout(0.03),
    Dense(10, activation="softmax")
], name="Model3_GELU_Final")
```

**Architecture (from interactive diagram):** Input (784) → Dense-600 (gelu) → Dense-512 (gelu) → Dense-256 (gelu) → Dense-10 (softmax)

**Complete breakdown of every new design decision:**

1. **GELU activation:** (see Cell 33) — smooth gradients, no dead neurons, better fine-tuning of class boundaries. Used in GPT/BERT.

2. **`kernel_initializer="he_normal"` (He/Kaiming initialization):**
   - Initializes weights from a normal distribution with std = √(2 / fan_in), where fan_in = number of inputs to that neuron.
   - Specifically designed for ReLU-family activations (including GELU). Ensures the signal flowing through layers maintains appropriate variance — not amplifying to infinity (exploding gradients) nor shrinking to zero (vanishing gradients).
   - **Trivia:** Named after Kaiming He (Microsoft Research, 2015). Before this, Xavier/Glorot initialization was standard (designed for sigmoid/tanh). He initialization enabled training ResNets with 100+ layers.

3. **`kernel_regularizer=l2(1e-5)` (L2 weight regularization / weight decay):**
   - Adds a penalty term to the loss: `total_loss = crossentropy + λ × Σ(w²)`, where λ = 1e-5.
   - Discourages large weights — large weights are a signature of overfitting (the model relies on specific weights to memorize examples).
   - λ = 1e-5 is deliberately tiny — just enough nudge without significantly constraining capacity.
   - **Combined strategy:** Using both small L2 (1e-5) *and* small Dropout (3–6%) is intentional — each addresses a different failure mode, and neither is strong enough alone to hurt capacity.

4. **Label smoothing = 0.03:**
   - Instead of hard targets `[0, 0, 0, 1, 0, ...]`, uses soft targets `[0.003, 0.003, 0.003, 0.973, 0.003, ...]`.
   - Prevents the model from becoming overconfident. Without smoothing, the loss rewards maximizing confidence toward "100% this is class X," which leads to poorly calibrated probabilities.
   - Especially useful on ambiguous classes — teaches the model that some uncertainty is appropriate, which improves generalization.
   - **Trivia:** Introduced in the Inception v3 paper (Szegedy et al., 2016); now standard in image classification, speech recognition, and NLP.

5. **Stratified train/val split:**
   - `stratify=np.argmax(y_train, axis=1)` guarantees the same class proportions in both the 90% training and 10% validation portions.
   - Without stratification, random splits might accidentally under-represent the "Shirt" class in training, producing a biased validation estimate.

6. **ReduceLROnPlateau callback:**
   - If `val_loss` hasn't improved for 2 consecutive epochs, multiply learning rate by 0.5 (minimum: 1e-6).
   - **Intuition:** The initial learning rate (1e-3) takes large steps appropriate for early learning. Near convergence, smaller steps allow finer weight adjustments without overshooting the optimal values. Analogous to using coarse sandpaper first, then fine sandpaper.
   - The learning rate is arguably the single most impactful hyperparameter in deep learning.

7. **EarlyStopping callback:**
   - If `val_categorical_accuracy` hasn't improved for 8 consecutive epochs, stop training and restore the best weights seen.
   - Prevents wasted compute and, critically, stops training before the model overfits in later epochs.
   - `restore_best_weights=True` ensures we always recover the historically best checkpoint even if the final epoch is worse.

**Result: 91% test accuracy — goal achieved!**

---

## Cell 36 — Model 3 Training

- Trains on the stratified split (`X_tr/X_val`), at most 30 epochs, all three callbacks active.
- Also saves `y_pred_classes3` and `report3` (per-class classification report as a dict) for use in the comparison dashboard cells.

---

## Cell 37 — Model 3 Training Curves

- **What to compare against Models 1 & 2:**
  - Training and validation accuracy curves should track each other much more closely — evidence of effective regularization.
  - A **vertical dashed red line** marks the epoch where early stopping triggered. If it stopped at epoch 20 instead of 30, the model converged and additional training would only hurt.
  - **ReduceLROnPlateau effect:** Brief "kinks" or small dips in the loss curve where the learning rate was halved — the optimizer momentarily takes smaller steps, often followed by a smoother descent.
- **The small overfitting gap is the key story:** the combination of L2, small Dropout, label smoothing, and early stopping compound to produce a more generalizable model despite its higher parameter count.

---

## Cell 41 — Final Evaluation & Misclassified Examples

- Evaluates `model` (which references `model3`) on the full 10,000-image test set.
- Produces the same classification report and confusion matrix as Model 1's evaluation, now for the final model.
- **Per-class outcomes:**

| Tier | Classes | F1 | Why |
|------|---------|-----|-----|
| Excellent | Trouser, Bag, Sandal, Ankle Boot, Sneaker | 97–99% | Visually distinct shapes |
| Good | Dress, T-shirt/top | 87–92% | Some overlap but generally distinguishable |
| Challenging | Pullover, Coat | 82–85% | Similar long-sleeved silhouettes |
| Hardest | Shirt | ~75% | Confused with T-shirt, Pullover, AND Coat simultaneously |

- **8 misclassified examples shown:**
  - Look at these images and ask: could *you* classify them correctly at 28×28 pixels?
  - A flat sneaker → predicted as Sandal (reasonable); an oversized shirt → predicted as Coat (reasonable); a minimalist dress → predicted as Shirt (reasonable).
  - **Key message:** These are not random errors — they are understandable confusions caused by genuine visual ambiguity at this resolution. The model has arguably reached the ceiling achievable with pixel-only MLP information.

---

## Cell 43 — Model Comparison Table

```python
df_comparison = pd.DataFrame(comparison_data)
```

- **What it shows:** 17-row, 3-column comparison table covering architecture, regularization, training setup, and final test accuracy for all three models.
- **How to read it:** Each column = one model. The highlighted "Test Accuracy" row at the bottom shows the final performance: Model 1 (~88.6%) → Model 2 (~89–90%) → Model 3 (91%).
- **Key narrative:** Model 3 is not dramatically larger than Model 2 — the accuracy gain came from smarter design (GELU, He init, L2, label smoothing, callbacks), not brute-force scaling.

---

## Cell 44 — All 3 Models Training Curves Overlay

*2-panel interactive chart: Validation Accuracy (left) | Validation Loss (right) — all 3 models on shared axes*

```python
fig.add_hline(y=0.90, line_dash="dash", annotation_text="90% target")
```

- **What it shows:** All three models' validation performance curves on the same axes. X-axis = epoch; Y-axis = val accuracy (left) or val loss (right). A horizontal dashed line marks the 90% accuracy target.
- **What each curve tells the story:**
  - **Model 1 (blue, 20 epochs):** Steady rise, plateaus ~88–89%. Clean but doesn't cross 90%.
  - **Model 2 (green, 60 epochs):** Rises faster initially but val accuracy plateaus and val loss climbs after ~20–30 epochs — **classic overfitting signature**. Long training without sufficient regularization.
  - **Model 3 (red, early-stopped):** Converges smoothly, crosses 90%, training and validation curves remain close — **healthy generalization signature**.
- **The 90% dashed line** immediately shows which models cleared the requirement: only Model 3 reliably stays above it.
- **(hover over individual data points to see exact val_accuracy per epoch for any model)**

---

## Cell 45 — Per-Class F1-Score Comparison (Grouped Bar Chart)

```python
fig.add_trace(go.Bar(name='Model 1 (Baseline)', x=df_perclass['Class'], y=df_perclass['Model 1 (Baseline)']))
fig.add_trace(go.Bar(name='Model 3 (GELU) ✨', ...))
```

- **What it shows:** For each of the 10 clothing classes, three bars side-by-side (one per model) showing the **F1-score** for that class. Y-axis ranges 0.5–1.0 so differences are visually amplified.
- **F1-score recap:** Harmonic mean of precision and recall. F1 = 1.0 = perfect; F1 = 0.5 = many errors in both directions. It's the best single-number metric when you care about both "making correct predictions" and "not missing true positives."
- **The 90% F1 dashed line** shows which class/model combinations cleared this per-class bar.
- **Key patterns:**
  - Footwear and Bag bars are tall for all three models — visually distinct shapes make them easy regardless of model sophistication.
  - Shirt bars are consistently the shortest — fundamentally ambiguous.
  - **Look for where Model 3's red bar is tallest vs. blue/green** — these classes benefited most from GELU and advanced regularization. Shirt and Pullover typically show the biggest improvement.
- **"Biggest Improvements" printed below:** Automatically identifies classes where Model 3 gained > 3% F1 points over Model 1.
- **(click legend to hide/show individual models for easier pairwise comparisons)**

---

## Cell 47 — Extract Hidden Layer Activations (Keras 3.x compatible)

```python
x = tf.constant(X_test, dtype=tf.float32)
for layer in model3.layers[:-1]:   # all layers except final Dense(10)
    x = layer(x, training=False)
hidden_activations = x.numpy()     # Shape: (10000, 256)
```

- **What this does:** Runs the 10,000 test images through every layer of Model 3 *except* the final `Dense(10)` softmax, capturing the 256-dimensional output of the last hidden layer as a numpy array.
- **Why this Keras 3.x approach:** The classic `Model(inputs=model.input, outputs=model.layers[-2].output)` sub-model API fails in Keras 3.x for Sequential models (`.input` and `.output` attributes aren't available until the model is called). The manual layer-by-layer forward pass works in all versions.
- **`training=False`:** Disables Dropout during this pass — at inference we want the full activation signal from all neurons, not the masked version used during training.
- **What `hidden_activations` represents:** Each row is a test image; each of the 256 columns is one "learned feature" — a combination of pixel signals that the model found useful for classification. These 256 values are not human-interpretable (they're linear combinations of thousands of weights), but they carry far more discriminative information than raw pixels.
- **Then UMAP projects 256D → 3D** using the same pipeline as earlier.

---

## Cell 48 — Raw Data vs. Learned Representations (Side-by-Side 3D UMAP)

*Two separate 3D scatter plots: raw pixel UMAP (first) | hidden layer activation UMAP (second)*

**"BEFORE" plot — Raw pixel UMAP:**
- Each of 10,000 test images represented by its 784 raw pixel values, compressed to 3D via PCA(50) → UMAP(3D).
- **Expected appearance:** Moderate cluster separation — Trouser, Bag, Sandal form identifiable islands; Shirt/T-shirt/Pullover/Coat form a heavily overlapping mass.
- This is what the model *starts with* as its raw material.

**"AFTER" plot — Hidden layer activation UMAP:**
- Each test image represented by its 256 learned feature values from Model 3's last hidden layer, compressed to 3D via UMAP.
- **Expected appearance:** Dramatically tighter, more separated clusters across all classes. Even the previously overlapping upper-body clothing classes should show more distinct structure.
- **Why they look different:** The 256 hidden layer values are not raw pixels — they are **learned representations**. Through training toward 91% accuracy, Model 3 transformed 784 raw pixel values into 256 "smart features" optimally useful for distinguishing the 10 classes.

**This is the single most powerful visualization in the notebook:**
- It provides **visual proof** that the model learned something meaningful — not just memorization of training images, but a structured internal representation that respects the underlying class geometry.
- It makes concrete what "feature learning" means in deep learning: the network built its own internal vocabulary for describing garments.
- The fact that even in the learned space some classes still overlap (chiefly Shirt) confirms this overlap is a fundamental property of the data — not a model failure that could be engineered away.

---

## Final Analysis — Overall Narrative Arc

**Progression Summary:**

| Iteration | Model | Key Technique Added | Test Accuracy |
|-----------|-------|---------------------|----------------|
| 1 | Baseline | ReLU + Dropout(0.3) | ~88.6% |
| 2 | Larger | More capacity, less Dropout, 60 epochs | ~89–90% |
| 3 | **Final** | GELU + L2 + He init + label smoothing + LR schedule + early stopping | **91% ✅** |

**Five takeaways to leave with:**

1. **Data understanding predicts model difficulty:** The EDA (mean images, PCA, UMAP) told us exactly which classes would be hard before building any model. Time spent understanding data is never wasted.

2. **The right activation function matters:** Swapping ReLU for GELU — a mathematical formulation change, not a structural one — meaningfully contributed to crossing 90% by eliminating dead neurons and smoothing the optimization landscape.

3. **Regularization is a portfolio strategy:** Model 3 uses four lightweight regularizers (Dropout, L2, label smoothing, early stopping) rather than one aggressive one. Each addresses a different failure mode; none limits capacity alone.

4. **Training dynamics are as important as architecture:** ReduceLROnPlateau and EarlyStopping are the mechanisms that actually bring a model to its best possible weights — they're not optional extras.

5. **There is a known ceiling:** The per-class F1 analysis shows Shirt is the fundamental bottleneck. No MLP, regardless of sophistication, will dramatically exceed this on Fashion MNIST's genuinely ambiguous classes. The natural next step — CNNs — preserve spatial structure and typically reach 93–95% on this dataset.

