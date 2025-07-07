**1.研究生毕业论文设计**

**2.肝血管分割任务，结合图神经网络**

![img.png](img.png)
基于多任务和血管先验引导的分割网络（MTPG-Net）框架

数据集使用3Dircadb1

SubeNet使用的checkpoint为 [SAM-Med3D-turbo](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing)


# Liver Vessel Segmentation with Graph (MTPG-Net)

A research project for liver vessel segmentation combining **multi-task learning**, **vessel prior guidance**, and **graph neural networks**, based on the **3Dircadb1** dataset.

This repository implements **MTPG-Net (Multi-Task and Prior-Guided Network)** for accurate hepatic vessel segmentation. It incorporates a graph learning module to model vascular connectivity and employs **SAM-Med3D-turbo** as the backbone for feature extraction.

<p align="center">
  <img src="img.png" alt="MTPG-Net Architecture" width="600"/>
</p>
