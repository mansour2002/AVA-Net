# AVA-Net
Deep Learning for Arterial-Venous-Area (AVA) Segmentation using OCTA images. The implementation is using Python and TensorFlow.

Overview
------------
In this project, we present a fully convolutional network (FCN), AVA-Net, a U-Net-like architecture, for fully automated arterial-venous area (AVA) segmentation using OCT-angiography (OCTA) images. The AVA-Net architecture is illustrated in below
![The AVA-Net](https://github.com/mansour2002/AVA-Net/blob/main/Figures/Slide2.PNG?raw=true)



Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used. 

Figures (A) and (B) show representative OCTA image and corresponding manually generated artery-vein (AV) map. For generating AVA maps, the k-nearest neighbor (kNN) classifier is used to classify background pixels as pixels in arterial or venous areas. The output of the kNN classifier is presented in figure (C) with a lighter tone of blue and red comparing to arteries and veins presented in figure (B). The union of the arteries and veins with corresponding arterial and venous areas generate the AVA maps represented in figure (D).
![The AVA-Net](https://github.com/mansour2002/AVA-Net/blob/main/Figures/Slide%201.png?raw=true)


Dependencies
------------
- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

Citations
------------
Mansour Abtahi, David Le, Jennifer I. Lim, and Xincheng Yao, "AVA-Net: an open-source deep learning network for arterial-venous area segmentation in OCT angiography," (2022)

