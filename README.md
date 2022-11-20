# Multimodal-Brain-Tumor-Segmenatation-BraTS-2020
This research work basically highlights my undergrad thesis works. In my thesis, I have worked on the BraTS 2020 dataset. My total journey of thesis from building various models to writing paper is presented here.
## Abstract 

<p align ="justify>

Brain tumors are classified as primary (originating in the brain) or secondary (metastasizing from elsewhere). Gliomas are the most prevalent malignant primary brain
tumor in adults, accounting for 80%.According to the WHO, gliomas are classified
into four grades: low grade (LGG) (grades 1-2), which are less prevalent and have
low blood concentration and sluggish growth, and high grade (HGG) (grades 3-4),
which have rapid growth and aggressiveness.</p>

<p align = "justify"> 
The edematous/invaded tissue surrounding the tumor, the necrotic core (filled
with fluid), and the enhancing and non-enhancing tumor (solid) core. T1-weighted,
postcontrast T1-weighted, and Fluid-Attenuated Inversion Recovery(FLAIR) MRI
modalities are extensively utilized to identify the diagnosis, therapy, and evaluation
of the disease because they reflect the diverse biological characteristics of the tumor.
These MRI techniques facilitate tumor analysis, but require the laborious and timeconsuming manual identification of tumor regions. As a result of deep learning models
in computer vision, approaches for the autonomous segmentation of tumor regions
have emerged. In an effort to automate the process of segmenting tumor regions, a
novel segmentation framework including efficient, contemporary deep learning blocks
is provided. Our proposed model is a cascaded encoder-decoder network with two
stages. In both stages of training, a variational autoencoder branch is included. In
addition, a transformer module is incorporated into the bottleneck layer to account
for long-range dependencies. Attention gate is incorporated in the second stage to
assist the network in segmenting smaller tumor patches. This block increases the
dice score for smaller sub-regions of glioma, such as the tumor that is enhancing.
Ultimately, our suggested technique is validated using the BRATS-2020 benchmark
dataset. Our method yields equivalent results in comparison to the standard methods. Specifically, 87.09 percent, 80.32 percent, and 74.63 percent dice scores are
obtained when segmenting the entire tumor (WT), tumor core (TC), and enhanced
tumor (ET), respectively. Ablation study is also undertaken to better comprehend
the generalization of the design. </p>

## Proposed Architecture

<img src = "https://github.com/PrasunDatta/Multimodal-Brain-Tumor-Segmenatation-BraTS-2020/blob/main/Brats1.PNG" align ="center" />

### Let's Look At Transformer Module

<img src = "https://github.com/PrasunDatta/Multimodal-Brain-Tumor-Segmenatation-BraTS-2020/blob/main/Brats2.PNG" align ="center" />

## Results

<img src ="https://github.com/PrasunDatta/Multimodal-Brain-Tumor-Segmenatation-BraTS-2020/blob/main/Brats3.PNG" align ="center" />

### Let's Look at Graphs

<img src ="https://github.com/PrasunDatta/Multimodal-Brain-Tumor-Segmenatation-BraTS-2020/blob/main/Brats%204.PNG" align ="center" />
<img src = "https://github.com/PrasunDatta/Multimodal-Brain-Tumor-Segmenatation-BraTS-2020/blob/main/Brats5.PNG" align = "center" />


