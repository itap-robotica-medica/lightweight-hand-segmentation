# Hand Segmentation
This repository contains the Python implementation of the algorithm for hand segmentation presented on "Lightweight Real Time 
Hand Segmentation Leveraging MediaPipe Landmark Detection".


# Overview
Real-time hand segmentation is a key process in applications that require human-computer interaction,
such as gesture recognition or augmented reality systems. However, the infinite shapes and orientations
that hands can adopt, their variability in skin pigmentation, and the self-occlusions that
continuously appear in images make hand segmentation a truly complex problem, especially with
uncontrolled lighting conditions and backgrounds. The development of robust, real-time hand segmentation
algorithms is essential to achieve immersive augmented reality and mixed reality experiences
by correctly interpreting collisions and occlusions. 

We present a simple but powerful algorithm based on the Media Pipe Hands solution, a highly optimized neural network. The algorithm
processes the landmarks provided by Media Pipe using morphological and logical operators to obtain
the masks that allow dynamic updating of the skin color model. Different experiments were carried out
comparing the influence of the color space on skin segmentation, with the CIELab color space chosen
as the best option. An average intersection over union (IoU) of 0.869 was achieved on the demanding
Ego2Hands dataset running at 90 frames per second on a conventional computer without any hardware
acceleration.


# Citation
If you use this repository or any of its components and/or our paper as part of your research, please cite the publication as follows:
> G. Sánchez-Brizuela *et al.* "Lightweight Real Time Hand Segmentation Leveraging MediaPipe Landmark Detection," *Virtual Reality*,  27, 3125–3132 (2023). https://doi.org/10.1007/s10055-023-00858-0

```
@article{handsegmentation2023,
  title={Lightweight Real Time Hand Segmentation Leveraging MediaPipe Landmark Detection},
  author={Sánchez-Brizuela, Guillermo and Cisnal, Ana and de la Fuente-Lopez, Eusebio and Fraile, Juan-Carlos and Pérez-Turiel, Javier},
  journal={Virtual Reality},  
  pages = {3125--3132},
  vol = {27},
  year={2023},
  doi = {10.1007/s10055-023-00858-0},
}
```
