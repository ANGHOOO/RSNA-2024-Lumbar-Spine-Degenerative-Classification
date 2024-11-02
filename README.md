# 🥉133rd Place Solution to the RSNA 2024 Lumbar Spine Degenerative Classification Competition

## Introduction

This repository includes the solution that achieved 133rd place in the competition, with private leaderboard scores of 0.47 and 0.52. My approach leverages a blend of algorithmic precision and memory-based techniques, utilizing Transformers as the core architecture to accommodate the unique needs of lumbar spine degeneration classification.

## Overview

This pipeline has several key components:

<img src="https://github.com/user-attachments/assets/873bccc2-c857-4e64-a2ce-67cba2ef99c7" width="600" alt="Model Architecture">

### Problem Foundation
https://www.kaggle.com/code/abhinavsuri/anatomy-image-visualization-overview-rsna-raids

The starting point for this work was inspired by Abhinav Suri’s notebook, which introduced a foundational analysis of 2D sagittal T1, axial T2, and other imaging methods for spinal pathologies.

### Slice Matching Issue

Due to inconsistencies in slice alignment across images, I implemented a flexible Transformer architecture to manage and integrate these inconsistencies effectively.

### 2D UNet for ROI Localization

The 2D UNet model helped identify regions of interest (ROI) across different slices. I utilized pre-trained models to capture detailed spatial features, which enhanced the detection of vertebral regions. With the imputed coordinates, I could pinpoint regions across sagittal and axial images for further analysis.

### CNN for Crop-Level ROI Localization

Initially, full-slice images were used to train the Transformer on multiple slice "sandwiches" per level, capturing a high-level overview. However, transitioning to targeted ROI localization using cropped slices offered better accuracy without downsampling. Each crop was then classified, enabling a more efficient Transformer encoding process.

### Data Augmentation & Label Imputation

To expand the dataset, I augmented Axial T2 slices through flipping and generated inferred labels for slices that lacked manual labels. Introducing this additional data improved the model’s ability to generalize across unlabeled regions, enhancing predictions for spinal pathologies.

## Scores

| Model       | Public LB | Private LB |
|-------------|-----------|------------|
| Transformer | 0.47      | 0.52       |

## Final Model - Vision Transformer (ViT) over Crops

I used a Vision Transformer (ViT) with ResNet18 as an encoder across crop sequences. Predictions for foraminal, spinal, and subarticular conditions were refined through cross-validation, balancing performance across levels.

## Inference & Training

Training involved five-fold cross-validation for robust results, with final predictions submitted via the provided script.

### Code structure

If you wish to dive into the code, the repository naming should be straight-forward. Each function is documented.
The structure is the following :

```
src
├── AxialT2_train.py # Train ViT segmentation model for Axial T2
│  
├── SagittalT1_train.py # Train ViT segmentation model for Sagittal T1
│
├── SagittalT2_train.py # Train ViT segmentation model for Sagittal T2
│
├── predict.py # Inference pipeline for kaggle submission by using above all trained models                         
```

## What Didn't Work & Future Work

### What Didn't Work

Despite achieving a respectable 133rd place in the RSNA 2024 competition, several challenges were encountered during the development of the model:

1. **Slice Alignment Issues**: The flexible Transformer architecture helped to some extent in managing inconsistencies in slice alignment; however, it did not completely eliminate the noise introduced by misaligned slices. This led to occasional inaccuracies in ROI detection, affecting overall performance.

2. **Limitations of 2D UNet**: While the 2D UNet model effectively identified regions of interest (ROI), it struggled with complex spinal pathologies that exhibit subtle variations across different imaging modalities. The model’s reliance on pre-trained weights may have restricted its ability to fully adapt to the unique characteristics of the dataset.

3. **Data Augmentation Impact**: Although data augmentation through flipping helped expand the dataset, the approach didn’t significantly enhance generalization. The generated inferred labels sometimes introduced noise, leading to incorrect classifications, especially in challenging cases where subtle variations in spinal structure were critical.

4. **Model Complexity**: The Vision Transformer (ViT) architecture, while powerful, required extensive computational resources. Training times were lengthy, and the model's complexity made it challenging to fine-tune hyperparameters effectively.

### Future Work

To improve upon the current solution, the following avenues for future work are proposed:

1. **Advanced Alignment Techniques**: Implementing more sophisticated slice alignment techniques, such as image registration methods, could enhance the consistency across different imaging slices, potentially reducing noise in the dataset.

2. **3D UNet Exploration**: Exploring a 3D UNet architecture could leverage the volumetric nature of medical images, providing a more comprehensive analysis of the spinal structures across all slices rather than relying solely on 2D representations.

3. **Enhanced Data Augmentation**: Investigating more advanced data augmentation techniques, such as synthetic data generation or using generative adversarial networks (GANs), might help improve the robustness of the model against variations in imaging data.

4. **Hyperparameter Optimization**: Utilizing automated hyperparameter tuning methods, such as Bayesian optimization or grid search, could refine the model's performance by systematically exploring optimal settings for various parameters.

5. **Transfer Learning and Ensemble Methods**: Implementing transfer learning from models pre-trained on similar medical imaging tasks or using ensemble methods could potentially enhance classification accuracy and robustness across different pathologies.

By addressing these areas, the overall performance of the classification model can be improved, paving the way for more accurate diagnostics in lumbar spine degenerative conditions.


