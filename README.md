# WeaklyNucleiSegmentation

## Weakly Supervised Deep Nuclei Segmentation with Sparsely Annotated Bounding Boxes for DNA Ploidy Analysis

### Abstract
The latent and high mortality of cancer promotes the research of prevention and early diagnosis. Cell ploidy status is a very
important parameter, which has the significance of diagnosis and prevention. DNA ploidy analysis of cells based on the image
cytometry technique and digital image processing is the dominant direction in the current study. Segmentation of the images is an
essential step but not a negligible problem because of the noise-filled background and intensity cluster with the nuclei and cells
overlapping. Typically, Such segmentation tasks require enormous professional labor costs to generate accurate pixel-wise labels for
fully supervised learning. In this study, we propose a weakly supervised segmentation framework based on sparse nuclear position
annotations. The framework is composed of two stages. During the first stage, we use the traditional image processing to extract
defective nuclear masks from partially labeled nuclei locations, which will be used to assist the parameter initialization of the model. In the second stage, we present a self-training strategy to exploit unlabeled nuclear regions to enhance nuclei detection and suppress false positives. The experimental results prove that our method can achieve a significant improvement on the accuracy of detection and provide precision enough segmentation mask to extract nuclear regions for DNA ploidy analysis

<p align="center">
  <img width="550" src="https://github.com/ZhiHuaYin/WeaklyNucleiSegmentation/blob/main/images/flowchart1.png" />
</p>

### Environment
* CUDA==10.0
* cuDNN==7.5
* caffe2

### Evaluate and Prediction

### Datasets
The dataset, which has more than 1000,000 nuclei in 23,485 images, was cropped from the Feulgen stained Whole Slide Images (WSIs). The number of targets in each image is distributed around 40, and the maximum number is 327. Compared to general datasets, the average number of targets per image is relatively large while the scale of targets is mainly concentrated below 32 × 32, which is a small size for the feature extractor block.

__Data distribution__
<p align="center">
  <img width="450" src="https://github.com/ZhiHuaYin/WeaklyNucleiSegmentation/blob/main/images/data_distribution%2B.png" />
</p>

__Initial segmentation annotations__
<p align="center">
  <img width="450" src="https://github.com/ZhiHuaYin/WeaklyNucleiSegmentation/blob/main/images/initial.png" />
</p>

