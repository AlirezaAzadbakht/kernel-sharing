
# Drastically Reducing the Number of Trainable Parameters in Deep CNNs by Inter-layer Kernel-sharing
## Abstract
Deep convolutional neural networks (DCNNs) have become the state-of-the-art (SOTA) approach for many computer vision tasks: image classification, object detection, semantic segmentation, etc. However, most SOTA networks are too large for edge computing. Here, we suggest a simple way to reduce the number of trainable parameters and thus the memory footprint: sharing kernels between multiple convolutional layers. Kernel-sharing is only possible between "isomorphic" layers, i.e. layers having the same kernel size, input and output channels. This is typically the case inside each stage of a DCNN. Our experiments on CIFAR-10 and CIFAR-100, using the ConvMixer and SE-ResNet architectures show that the number of parameters of these models can drastically be reduced with minimal cost on accuracy. The resulting networks are appealing for certain edge computing applications that are subject to severe memory constraints, and even more interesting if leveraging "frozen weights" hardware accelerators. Kernel-sharing is also an efficient regularization method, which can reduce overfitting.
  

## Add New Model

Models can be registered in `ConvMixer/models/convmixer.py`

  

## Training

For training the ConvMixer model on the Cifar-10 or Cifar-100 dataset, you can execute `cifar10_run.sh` or `cifar100_run.sh`
