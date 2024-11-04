# Evaluation metrics

### Methods

- [`torch-fidelity`](https://pypi.org/project/torch-fidelity/) library provides a suite of evaluation metrics for image quality, that measure the difference between two sets of images.
- [`lpips`](https://pypi.org/project/lpips/) library is designed for comparing individual images based on perceptual similarity, meaning it evaluates how similar two images look to the human eye.


### Table of contents

1. [Inception Score (IS)](#1-inception-score-is)
2. [Fréchet Inception Distance (FID)](#2-fréchet-inception-distance-fid)
3. [Kernel Inception Distance (KID)](#3-kernel-inception-distance-kid)
4. [Maximum Mean Discrepancy (MMD)](#4-maximum-mean-discrepancy-mmd)
5. [Precision and Recall for Distributions (PRD)](#5-precision-and-recall-for-distributions-prd)
6. [Learned Perceptual Image Patch Similarity (LPIPS)](#learned-perceptual-image-patch-similarity-lpips)


## Inception Score (IS)

- **Description**: The Inception Score evaluates the quality and diversity of images within a single set (often used for evaluating generated images).
- **How it Works**: 
  - Images are passed through a pre-trained Inception network (usually Inception-v3), which generates class probabilities.
  - The score considers two aspects: 
    1. High-quality images should have distinct class labels (i.e., high confidence for one class).
    2. There should be a diversity of classes across the set.
  - IS is calculated as the exponential of the Kullback-Leibler divergence between the conditional label distribution (given an image) and the marginal label distribution (across all images).
- **Output Interpretation**: Higher IS values indicate better quality and diversity in generated images, but the metric does not compare two sets of images. This makes IS less suitable for comparing real and generated datasets.

## Fréchet Inception Distance (FID)

- **Description**: Fréchet Inception Distance is a metric that compares two datasets (e.g., real and generated images) based on the similarity of their feature distributions.
- **How it Works**:
  - Images from both datasets are passed through the Inception network to extract feature representations.
  - These features are treated as multivariate Gaussian distributions, and the FID score calculates the Fréchet distance (a type of Wasserstein distance) between the two distributions.
  - FID considers both the mean and covariance of the feature distributions, so it’s sensitive to differences in both quality and diversity between the two sets.
- **Output Interpretation**: Lower FID scores indicate closer similarity between the datasets, with 0 meaning perfect similarity. Generally, FID scores below 50 are considered good, and scores below 10 are excellent.

## Kernel Inception Distance (KID)

- **Description**: Kernel Inception Distance is another similarity measure between two datasets, similar to FID, but it uses different statistical techniques.
- **How it Works**:
  - Like FID, KID extracts feature representations from an Inception network for images in each dataset.
  - Instead of calculating the Fréchet distance, KID calculates the squared Maximum Mean Discrepancy (MMD) between the feature representations.
  - KID uses a polynomial kernel function to compute MMD, which makes it more robust than FID when comparing small datasets.
- **Output Interpretation**: Lower KID values indicate higher similarity between the two sets. KID has the added advantage of being unbiased for small datasets, while FID can sometimes produce biased results with small sample sizes.

## Maximum Mean Discrepancy (MMD)

- **Description**: Maximum Mean Discrepancy is a general statistical test for comparing two probability distributions, used here to measure similarity between two sets of images.
- **How it Works**:
  - MMD compares the feature distributions of the two datasets by calculating the mean difference between the feature embeddings (from an Inception network) using kernel functions (such as Gaussian or polynomial kernels).
  - MMD is a non-parametric test, meaning it doesn’t assume the distributions are Gaussian like FID does. This flexibility allows MMD to capture more nuanced differences between the datasets.
- **Output Interpretation**: Lower MMD values indicate that the feature distributions are more similar. Like FID and KID, this means the generated and real images are closer in quality and diversity.

## Precision and Recall for Distributions (PRD)

- **Description**: PRD is a metric pair that provides both precision and recall values to assess the quality (precision) and diversity (recall) of generated images compared to a reference set.
- **How it Works**:
  - PRD assesses whether the generated images cover the distribution of real images (recall) and whether they maintain distinct quality (precision).
  - The metric divides the feature space into bins and calculates the overlap between the real and generated distributions.
  - Precision measures how well the generated data samples the reference distribution, while recall measures how well the real distribution covers the generated data.
- **Output Interpretation**: PRD returns two values:
  - **Precision**: Higher precision indicates that the generated images are of high quality and align well with real images.
  - **Recall**: Higher recall suggests that the generated set captures a wide diversity similar to the real dataset.
- **Applications**: PRD is particularly useful when both quality and diversity are essential, such as in generative models where we want generated images to both look realistic and cover a range of styles or classes.

## Learned Perceptual Image Patch Similarity (LPIPS)

- **Description**: LPIPS is a metric designed to measure the perceptual similarity between two individual images. It uses deep learning-based feature embeddings to evaluate how similar two images appear to the human eye.
- **How it Works**:
  - LPIPS uses a pre-trained neural network (e.g., AlexNet, VGG, or SqueezeNet) to extract high-level feature maps from both images.
  - The metric computes the distance between these feature maps, with the idea that images perceived as similar will have similar feature maps.
  - LPIPS applies a linear weighting to the feature maps, trained to align with human perceptual judgments of image similarity.
  - By comparing features at multiple levels, LPIPS captures both fine-grained details and overall structural similarity.
- **Networks**:
  - LPIPS offers three choices for feature extraction: **AlexNet**, **VGG**, and **SqueezeNet**.
  - **AlexNet**: Fast, effective for general perceptual similarity. **VGG**: More detailed, good for fine-grained similarity. **SqueezeNet**: Lighter model, often used for efficiency.
- **Output Interpretation**:
  - LPIPS returns a single scalar value representing the perceptual similarity between the two images.
  - **Lower values** indicate higher similarity (i.e., the images are perceived as more similar).
  - **Higher values** indicate greater dissimilarity.
  - Typically, an LPIPS score close to `0` suggests that the images are almost indistinguishable perceptually.
- **Applications**:
  - LPIPS is widely used for evaluating image quality in generative tasks (e.g., GANs) and super-resolution tasks.
  - It is particularly useful when the goal is to generate images that are visually close to a reference image from a human perspective, even if pixel-wise differences are present.
