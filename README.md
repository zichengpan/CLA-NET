---

<div align="center">    
 
# Contrastive Lie Algebra Learning for Ultra-Fine-Grained Visual Categorization

Xiaohan Yu*, Zicheng Pan*, Yang Zhao, Qin Zhang, and Yongsheng Gao

</div>

## Abstract
Ultra-fine-grained visual classification (ultra-FGVC) targets at classifying sub-grained categories of fine-grained objects. This inevitably requires discriminative representation learning within a limited training set. Exploring intrinsic features from the object itself via contrastive learning has demonstrated great progress towards learning discriminative representation. Yet forcingly dividing highly similar categories at the representation level may over-guide the learned feature space, leading to overfitting in the ultra-FGVC tasks. To this end, this paper introduces CLA-Net, a novel contrastive Lie algebra learning framework to address this fundamental problem in ultra-FGVC. The core design is a self-supervised module that performs self-shuffling and masking and then distinguishes these altered images from other images at a second-order representation level. This drives the model to learn an optimized feature space that has a large inter-class distance while remaining tolerant to intra-class variations. By incorporating this self-supervised module, the network acquires more knowledge from the intrinsic structure of the input data, which improves the generalization ability without requiring extra manual annotations. CLA-Net demonstrates strong performance on eight publicly available datasets, demonstrating its effectiveness in the ultra-FGVC task.

## Pre-trained ResNet50 Model Preparation
The pre-trained ResNet50 model can be download via: [this URL](https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/resnet/resnet50-19c8e357.pth) and be placed in the "pretrained_models" folder.
