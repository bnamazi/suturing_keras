This repository contains the source code for the video-based assessment of open suturing and knot-tying. Two models were developed to identify the incorrect instrument holding and incorrect knot-tying in short videos taken during medical students training. The models are implemented using Keras.

Paper: TBD


Correct Pickup Holding           |  Incorrect Pickup Holding
:-------------------------:|:-------------------------:
![](./images/correct%20pickup%20holding.jpg)  |  ![](./images/incorrect%20pickup%20holding.jpg)

The block diagram of our instrument holding error detection model:

<h1 align="center">
  <img src="./images/instrument holding model bd.png">
</h1>

The block diagram of our knot-tying holding error detection model:

<h1 align="center">
  <img src="./images/knot-tying model bd.png">
</h1>

## TODO

test script - jupyter notebook

Data preparation


## References
X3d: https://github.com/fcogidi/X3D-tf

video data generator: https://github.com/metal3d/keras-video-generators