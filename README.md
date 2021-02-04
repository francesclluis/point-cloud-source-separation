Music source separation conditioned on 3D point clouds
====

[Paper](https://arxiv.org/abs/2102.02028) by Francesc Lluís, Vasileios Chatziioannou, Alex Hofmann

## Abstract
Recently, significant progress has been made in audio source separation by the application of deep learning techniques. Current methods that combine both audio and visual information use 2D representations such as images to guide the separation process. However, in order to (re)-create acoustically correct scenes for 3D virtual/augmented reality applications from recordings of real music ensembles, detailed information about each sound source in the 3D environment is required. This demand, together with the proliferation of 3D visual acquisition systems like LiDAR or rgb-depth cameras, stimulates the creation of models that can guide the audio separation using 3D visual information. This paper proposes a multi-modal deep learning model to perform music source separation conditioned on 3D point clouds of music performance recordings. This model extracts visual features using 3D sparse convolutions, while audio features are extracted using dense convolutions. A fusion module combines the extracted features to finally perform the audio source separation. It is shown, that the presented model can distinguish the musical instruments from a single 3D point cloud frame, and perform source separation qualitatively similar to a reference case, where manually assigned instrument labels are provided.

![diagram](img/diagram.png)

## Code

Coming soon
