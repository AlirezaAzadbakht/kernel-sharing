
# Drastically Reducing the Number of Trainable Parameters in Deep CNNs by Inter-layer Kernel-sharing
## Paper abstract:
Deep convolutional neural networks (DCNNs) have become the state-of-the-art (SOTA) approach for many computer vision tasks: image classification, object detection, semantic segmentation, etc. However, most SOTA networks are too large for edge computing. Here, we suggest a simple way to reduce the number of trainable parameters and thus the memory footprint: sharing kernels between multiple convolutional layers. Kernel-sharing is only possible between "isomorphic" layers, i.e. layers having the same kernel size, input and output channels. This is typically the case inside each stage of a DCNN. Our experiments on CIFAR-10 and CIFAR-100, using the ConvMixer and SE-ResNet architectures show that the number of parameters of these models can drastically be reduced with minimal cost on accuracy. The resulting networks are appealing for certain edge computing applications that are subject to severe memory constraints, and even more interesting if leveraging "frozen weights" hardware accelerators. Kernel-sharing is also an efficient regularization method, which can reduce overfitting.
  
<!-- ![alt text](https://github.com/AlirezaAzadbakht/kernel-sharing/blob/main/figs/shared-kernel.png?raw=true=100x20) -->
<p align="center">
  <img src="https://github.com/AlirezaAzadbakht/kernel-sharing/blob/main/figs/shared-kernel.png" style="width:40%;" />
</p>

## Add new model

New models with the desired configuration could be registered in `ConvMixer/models/convmixer.py` and `SEResNet/models/se_resnet.py`
  
## Instal requirements 
To prepare suitable envirments, execute `pip install -r requirements.txt` in a conda enviorments.

## training

For training the ConvMixer model on the Cifar-10 or Cifar-100 dataset, you can execute `convmixer_cifar10_run.sh` or `convmixer_cifar100_run.sh`
 and for the SE-ResNet, you can execute `seresnet_cifar10_run.sh` or `seresnet_cifar100_run.sh`
