# AVA-Net
Deep Learning for Arterial-Venous-Area (AVA) Segmentation using OCTA images. The implementation is using Python and TensorFlow.

Overview
------------
In this project, we present a fully convolutional network (FCN), AVA-Net, a U-Net-like architecture, for fully automated arterial-venous area (AVA) segmentation using OCT-angiography (OCTA) images. The AVA-Net architecture is illustrated in below


(https://github.com/mansour2002/AVA-Net/blob/main/Figures/Slide2.PNG?raw=true)

Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All en face OCT/OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used. The en face OCT was generated as a 3D projection of the retina slab. After image reconstruction, both en face OCT and OCTA were exported from ReVue software interface (Optovue) for further processing.

Network Architecture
------------
The MF-AV-Net is an FCN based on a modified UNet algorithm, which consists of an encoder-decoder architecture. The input to the MF-AV-Net can be of a single channel or a two-channel image. The network architecture presented below represents a late fusion approach that combines the outputs of two networks trained on different imaging modalities, OCT and OCTA, respectively.

![The late stage fusion approach of MF-AV-Net, which employs different expert networks for OCT and OCTA, and combines the output of the two networks.](https://github.com/dleninja/multimodal-avnet/blob/main/misc/figure_Late_fusion.png?raw=true)

Dependencies
------------
- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

Citations
------------
Mansour Abtahi, David Le, Jennifer I. Lim, and Xincheng Yao, "MF-AV-Net: an open-source deep learning network with multimodal fusion options for artery-vein segmentation in OCT angiography," Biomed. Opt. Express 13, 4870-4888 (2022) https://doi.org/10.1364/BOE.468483

