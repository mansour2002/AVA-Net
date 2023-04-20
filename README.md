# AVA-Net
Deep Learning for Arterial-Venous-Area (AVA) Segmentation using OCTA images. The implementation is using Python and TensorFlow.


Overview
------------
In this project, we present a fully convolutional network (FCN), AVA-Net, a U-Net-like architecture, for fully automated arterial-venous area (AVA) segmentation using OCT-angiography (OCTA) images. The AVA-Net architecture is illustrated below
![The AVA-Net](https://github.com/mansour2002/AVA-Net/blob/main/Figures/Slide2.PNG?raw=true)



Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used. 

Figures (A) and (B) show a representative OCTA image and corresponding manually generated artery-vein (AV) map. For generating AVA maps, the k-nearest neighbor (kNN) classifier is used to classify background pixels as pixels in arterial or venous areas. The output of the kNN classifier is presented in Figure (C) with a lighter tone of blue and red compared to arteries and veins presented in Figure (B). The union of the arteries and veins with corresponding arterial and venous areas generates the AVA maps represented in Figure (D).
![The AVA-Net](https://github.com/mansour2002/AVA-Net/blob/main/Figures/Slide%201.png?raw=true)


Dependencies
------------
- Tensorflow >= 1.31.1
- Keras >= 2.2.4
- Python >= 3.7.1


To obtain the trained weights of AVA-Net, please feel free to contact the corresponding author via email (Prof. Xincheng Yao, xcy@uic.edu), and tell us about your study. We can provide you with the necessary information and resources.


Paper Link: https://www.nature.com/articles/s43856-023-00287-9


Citations
------------
Mansour Abtahi, David Le, Behrouz Ebrahimi, Albert K. Dadzie, Jennifer I. Lim & Xincheng Yao, "An open-source deep learning network AVA-Net for arterial-venous area segmentation in optical coherence tomography angiography". Communications Medicine, 3, 54 (2023). https://doi.org/10.1038/s43856-023-00287-9

