# Go One Dimension Higher: Can Neural Networks for Point Cloud Analysis Help in Image Recognition?

> [color=#e0bf55] Zhaiyu Chen and Qian Bai
> [color=#e0bf55] [time=Tue, Jun 30, 2020 9:20 PM]

![](https://i.imgur.com/zDdbQei.png =x)

---

## Introduction

Compared to images, point clouds are unordered and anisotropically distributed in space, which makes them difficult to process using traditional 2D convolutional neural networks (CNNs). Recent deep learning architectures like PointNet[^1] can directly consume raw point clouds for semantic labelling. Such networks also preserve the permutation invariance property of point clouds, which is not present in CNNs for images.

Inspired by the sanity check made by Qi et al.[^1] using PointNet, which treats an image from MNIST dataset as a 2D point cloud, in this project we discuss whether neural networks designed for point cloud analysis can help image recognition. 

> *While we focus on 3D point cloud learning, a sanity check experiment is to apply our network on a 2D point clouds - pixel sets [^1].*

In this project, we employ three deep neural networks originally for point cloud analysis, namely PointNet, PointNet++ and PointCNN. These networks exhibit excellent performance on point cloud classification and segmentation, but we wonder its adaptivity on image recognition. That being said, we have no ambition to defeat the state-of-the-art CNNs. Instead, the goal of this project is to discover the potential of these networks designed for 3D point set on 2D pixel set.

The main contributions of our project can be summarized as follows:

* We implemented a unified PyTorch framework for image classification, based on the existing code on [PointNet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and the [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) library. To our best knowledge, no previous work has implemented this. Our code is open-sourced at [Github](https://github.com/chenzhaiyu/pointset-image-recognition).
* We extended the datasets used in the original sanity check for PointNet, PointNet++ and PointCNN ---MNIST and CIFAR10--- by one additional dataset, Fashion MNIST.
* We analysed the impact of shape information in the classification by applying GrabCut on CIFAR10 dataset to distil the foreground shape.
* We discussed whether PointCNN shows the characteristic of shift invariance.
* We extended the usecase of PointNet and PointNet++ to image semantic segmentation.

## Related Work

### Image Recognition

Image recognition has been deployed by many industries and already reached the mainstream. Many visual recognition tasks, if not all, have benefited from CNNs, which progressively extract higher- and higher-level representations of image contents. Instead of using handcrafted features like textures and shapes, a CNN takes an image's raw pixels as input and learns the optimal feature representation, and ultimately infers the image semantics.

For example, VGG16 is a CNN model proposed by K. Simonyan and A. Zisserman [^6]. The model achieves a top-5 test accuracy of 92.7% in ImageNet --- a dataset of over 14 million images belonging to 1000 classes. *Figure 1* illustrates the architecture of VGG16. Besides VGG16, other CNNs are used in this project as baselines including Maxout Network[^8], ResNet[^7], etc.

![](https://i.imgur.com/I0DXDir.png =600x)
*Figure 1: VGG16 Architecture[^13]*

### Point Cloud Analysis

Recently, point cloud, as an important source of 3D data, is exploited in the deep learning community. Point cloud data is inherently embedded in 3D space and has rich semantic information. Next we will introduce the three neural networks employed in this project, namely PointNet, PointNet++ and PointCNN.

#### PointNet

PointNet exploits features of a point cloud by extracting a global signature from all points, which substantially neglect the local embedding of the point features[^1]. *Figure 2* shows the architecture of PointNet.

![](https://i.imgur.com/tFMf1iQ.png =700x)
*Figure 2: PointNet Architecture[^1]*

The input points are first aligned by multiplying a transformation network called T-Net, which resembles a mini PointNet and encodes the geometric transformation upon the point set by learning the conversion matrix to ensure the invariance of the model. After extracting the features of each point cloud through multiple multi-layer perceptrons (MLPs), a T-Net is used again to align the features. Then max-pooling operations are performed on each dimension of the feature to get the final global feature. For classification tasks, the global features are fed into an MLP to predict the final classification score; for segmentation tasks, the global feature is concatenated with the previously learned local features of each point to encode the per-point features, and then the classification results of each data point are obtained by the stacked MLPs.

#### PointNet++

Inspired by CNN's layer-by-layer abstraction of local features, PointNet++ extracts local features at different scales and obtain deep features through a multi-layer network structure[^2]. *Figure 3* shows the architecture of PointNet++.

![](https://i.imgur.com/sZxSmRK.png =700x)
*Figure 3: PointNet++ Architecture[^2]*

There are three key components in PointNet++, namely sampling, grouping and feature learning. Farthest point sampling (FPS) is performed to sample the data points, which can better cover the entire sampling space compared with random sampling. The "locality" of a point is partially composed of other points in the spherical space drawn by a given radius around it, which facilitate the subsequent feature extraction for each local spatial neighbourhood. PointNet provides a feature extraction network based on point cloud data, which is directly applied to grouped points to extract local features.

The above components constitute the basic processing module of PointNet++. If multiple such processing modules are cascaded together, PointNet++ can get deep semantic features from shallow features. For segmentation tasks, it is also necessary to upsample the features after downsampling, so that each point in the original point cloud is assigned with corresponding features.

#### PointCNN

For a typical convolution operation, the output changes with the order of the input points, which is an undesired property. Therefore, the input order of the points in a point cloud is the main problem that hinders the operation of convolution. To address the issue, PointCNN defines a transformation matrix $\mathcal{X}$, which can process the input points with an arbitrary order to obtain a feature that is order-independent. *Figure 4* details the $\mathcal{X}$-Conv operator, where $\textbf{P}=(p_1, p_2, ..., p_k)^T$ is the neighbouring points, with $\textbf{F}=(f_1, f_2, ..., f_k)^T$ as their features. $\textbf{K}$ represents the convolution kernel and $p$ represents the representative points. The operation can also be expressed more concisely as

\begin{equation}
\textbf{F}_p = \mathcal{X} - Conv(\textbf{K}, p, \textbf{P}, \textbf{F}) = Conv(\textbf{K}, MLP(\mathcal{X}-p) \times [MLP_{\rho}(\mathcal{X}-p), \textbf{F}].
\end{equation}

![](https://i.imgur.com/KkMhGfV.png =670x)
*Figure 4: $\mathcal{X}$-Conv operator[^3]*


With the $\mathcal{X}$-Conv operation as building blocks, PointCNN is able to perform convolutions on the unordered point set, similar to that of 2-dimension grids, as illustrated in *Figure 5*.

![](https://i.imgur.com/yyNo6L4.png =600x)
*Figure 5: Hierarchical convolution on regular grids (upper) and point clouds (lower)[^3]*

### Graph Cut and GrabCut

Graph cut, as an energy minimization technique, can be utilised to solve many low-level computer vision problems. Let $G = <N, L>$ be a graph formed by a set of nodes $N$ and a set of directed links $L$ that connect them. In $N$ there are two special *terminal* nodes which are defined as the *source* $S$ and the *sink* $T$, and the other *non-terminal* nodes $P$. This kind of graph is called $s-t$ graph.

Image segmentation is generally regarded as a pixel labelling problem. For binary segmentation, two labels are to be assigned corresponding to pixels belonging to the background and the object/foreground. Inherently, the binary segmentation can be represented as a partition of the $s-t$ graph, which can be solved by graph cut operation where the two subsets $S$ and $T$ represent the foreground and background respectively, as illustrated in *Figure 6*.

![](https://i.imgur.com/kftr0jH.png =600x)
*Figure 6: Image segmentation as a graph cut problem*

GrabCut is an image segmentation method based on graph cut, where minimal user interaction is needed compared to the original graph cut method[^5]. Starting with a user-specified bounding box around the object to be segmented, the algorithm uses a Gaussian mixture model to approximate the colour distribution of the target object and that of the background. Taken from the bounding box, the segmentation can be further improved with user-assigned foreground/background pixels. This two-step procedure is repeated until convergence. *Figure 7* shows an example illustrating the input and output of GrabCut.

![](https://i.imgur.com/EiFXaBq.jpg)
*Figure 7: An example of GrabCut[^14]*

Though designed for interactive segmentation, the process of GrabCut can be simply automated by only providing the bounding box covering the entire image, in which, however, the segmentation accuracy is expected to be significantly degraded.

## Configurations

### Hardware
* GPU: NVIDIA GTX 1660 with 6 GB VRAM
* CPU: Intel i5-9400F
* Memory: 16 GB

### Requirements
* Ubuntu 18.04
* PyTorch 1.2
* PyTorch Geometric 1.5
Please refer to the [environment.yml](https://github.com/chenzhaiyu/pointset-image-recognition/blob/master/environment.yml) for a complete list of required packages.

## Datasets

* **MNIST:** a database of handwritten digits from 0 to 9, containing a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28×28 gray-scale image. *Figure 8* shows some examples of the digits.<br>
![](https://i.imgur.com/N6er1JV.png =450x)
*Figure 8: Some examples in MNIST[^15]* 

* **Fashion MNIST:** a direct extension of MNIST, which shares the same image size and structure of training and testing splits with MNIST. The dataset includes 10 classes associated with fashion -- T-shirt/top, 	trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot, as shown in *Figure 9*.<br>
![](https://i.imgur.com/qYh8pgH.png =450x)
*Figure 9: A visualization of part of examples in Fashion MNIST, with each class taking three rows[^16]*

* **CIFAR10:** a dataset consisting of 60,000 32x32 colour images in 10 classes, which are split into 50,000 training examples and 10,000 test examples. Specifically, the classes contained in the dataset are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck, as shown in *Figure 10*. <br>
![](https://i.imgur.com/lfKmgWj.png =450x)
*Figure 10: Random examples of CIFAR10[^17]*

## Data Pre-processing: Convert Images to Point Clouds

* **MNIST/Fashion MNIST:** Both datasets contain gray-scale images. Pixels having 0 value indicate the background. Thus, we can extract the foreground pixels by simply setting a threshold of 0. To further convert an image into a 2D point cloud, we follow the sampling method used by Qi et al. for PointNet[^1]. Set size of 256 is used. If there are more than 256 foreground pixels in the set, we randomly sub-sample it; if there are less, we pad the set with the last pixel in the set. Then the output point cloud includes a series of (*x*, *y*, *z*) coordinates, with (*x*, *y*) assigned by (*row*, *col*) of each pixel and *z* = 0.0.

* **CIFAR10:** Images in CIFAR10 dataset have three channels with R, G, B features. The explicit shape information of the foreground object is "lost" when only looking at the pixel values. In this case, we randomly sample 512 pixels from the whole image. Beside (*x*, *y*, *z*), we also add R, G, B values of each pixel as three additional features in the output point cloud. *Figure 11* shows the examples. Code for data conversion can also be found in the [data_util](https://github.com/chenzhaiyu/pointset-image-recognition/tree/master/data_utils) directory. <br>
![](https://i.imgur.com/pcyLBJm.png =80x) ->![](https://i.imgur.com/IuI1Kmm.png =120x)->![](https://i.imgur.com/vvBFste.png =120x)
![](https://i.imgur.com/Pz8EPE4.png =80x) ->![](https://i.imgur.com/HKHDIaQ.png =120x)->![](https://i.imgur.com/fmgUFjV.png =120x)
![](https://i.imgur.com/uLThjTG.png =80x) ->![](https://i.imgur.com/o87TvB8.png =120x)->![](https://i.imgur.com/YlPtY2A.png =120x)
![](https://i.imgur.com/sZ5ACBx.png =80x) ->![](https://i.imgur.com/PhtQjYR.png =120x)->![](https://i.imgur.com/c23bx0S.png =120x)
*Figure 11: Examples of converting images to point clouds. Left: Original images in CIFAR10[^17]. Middle: Point clouds with all pixels preserved. Right: Point clouds with 512 pixels randomly sampled.*

## Hyperparameter Settings
* Optimizer: Adam
* Learning Rate: 0.001
* Learning Rate Scheduler
    * Step Size: 20
    * Decay Rate: 0.7
* Number of Input Points
    * MNIST: 256
    * FASHION MNIST: 256
    * CIFAR10 (without GrabCut): 512
    * CIFAR10 (with GrabCut): 256


## Experiments and Results 

### Image Classification

First, we train PointNet, PointNet++ and PointCNN for image classification on MNIST and Fashion MNIST datasets, with results summarized in *Table 1*. It can be noticed that the reproduced results on MNIST using PointNet and PointNet++ are similar to that of the author's sanity check. In the case of PointCNN, the reason why we achieve much worse accuracy is arguable that we use a simple PyTorch implementation of PointCNN without full-fledged data augmentation used by the original paper.

|  Network |   MNIST  |Fashion MNIST|
| :--------| :------: |   :------:  |
| PointNet |   99.21 (99.22)|    84.75    |
|PointNet++|   99.52 (99.49) |    84.64    |
| PointCNN |   98.75 (99.54) |    77.53    |
| 2 Conv+3 FC ~ 500K parameters\*|99.40|93.40|
| 3 Conv+pooling+BN\*|  99.40 |90.30|
| ResNet18[^7]\*|   97.90  |    94.90   |
| Maxout[^8]\*  |   99.55  |      -     |
| Network in Network[^9]\* | 99.53| -   |
*Table 1: Test accuracy (%) for image classification on MNIST and Fashion MNIST. Accuracies shown inside the brackets are obtained by the original papers. Results of 2D CNN baselines are indicated with \*.*

From our results, PointNet++ shows a slight improvement over PointNet on MNIST dataset, which can be attributed to its hierarchical structure. For Fashion MNIST, however, the effect of such a structure is not apparent anymore. 

We also compare the classification results with some existing baseline using 2D CNNs (summarized in [Fashion-MNIST Benchmark](https://github.com/zalandoresearch/fashion-mnist#benchmark) and [Classification datasets results](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)), as shown with \* in *Table 1*. When classifying MNIST dataset, networks for point clouds show comparable results with 2D CNNs. However, this is not the case for Fashion MNIST. When the image becomes more complex, 2D CNNs win undoubtedly.

*Table 2* summarizes the results for classifying CIFAR10 images containing objects with diverse appearance (e.g. style, viewpoint, etc.) and additional RGB features. In this case, 2D CNNs\* achieve much better classification results than networks for point cloud analysis.

|  Network | CIFAR10  |
| :--------| :------: |
| PointNet | 11.02    |
|PointNet++| 10.19 (10.00) |
| PointCNN | 71.06 (80.22) |
| Maxout[^8]\*| 90.65  |
| Network in Network[^9]\*|    91.20    |
| ResNet110[^7]\*| 93.57 |
*Table 2: Test accuracy (%) for image classification on CIFAR10. Accuracies shown inside the brackets are obtained by the original papers. Results of 2D CNN baselines are indicated with \*.*

PointNet and PointNet++ completely fail on this task, with the accuracy (around 10%) no better than random choice. Considering that CIFAR10 misses explicit shape information of objects where the background and foreground in one image are indistinguishable without a clear threshold pixel value (e.g., 0 in MNIST and Fashion MNIST), it is challenging for networks like PointNet and PointNet++ to process the images. Another suspicion is that in PointNet and PointNet++, the RGB features become in-discriminative after being processed by the max-pooling.

As for PointCNN, the $\mathcal{X}$-Conv mechanism used in the architecture helps to aggregate local information in a CNN fashion. Features are efficiently exploited, resulting in reasonably much higher accuracy on CIFAR10.

### Impact of Shape Information

As revealed in the results of CIFAR10 classification (*Table 2*), the loss of shape information may hinder the learning of PointNet and PointNet++. This leads us to wonder: Will the networks be smarter if we manually extract the foreground from original CIFAR10 images? While checking the results in *Table 3*, the answer should be "yes".

After applying GrabCut to the original image dataset, both PointNet and PointNet++ have a significant improvement compared to the results without Grabcut. Here PointNet++ also shows an advantage over PointNet. with the comparison in *Table 3*, we can draw the following conclusion:

*RGB features alone are not sufficiently learnable by PointNet and PointNet++, which instead rely heavily on the explicit shape information.*
 
| Network    | CIFAR10 | CIFAR10+Grabcut |
|:--------| :------: | :------: |
| PointNet | 11.02    | 32.37    |
|PointNet++| 10.19    | 34.89    |
*Table 3: Test accuracy (%) for CIFAR10 classification without/with GrabCut*

Another note should be made on GrabCut. Since we automate the interactive process by only providing the bounding box covering the entire image, the segmentation performance is largely compromised. It turns out a considerable amount of the extracted objects are false or even blank if there is no conspicuous contrast between the background and the foreground (*Figure 12*).

![](https://i.imgur.com/beqfAzR.png =130x)![](https://i.imgur.com/IiarcWZ.png =130x) &nbsp; ![](https://i.imgur.com/bPvydWw.png =130x)![](https://i.imgur.com/8wa3E5Q.png =130x)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (a) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (b)
*Figure 12: (a) A successful example and (b) a "failed" example of GrabCut*

Although we try to mitigate this problem by only keeping the extracted image with more than 10% pixels of the original image, the classification results are limited by the GrabCut step.

### Is PointCNN Shift Invariant? 

![](https://i.imgur.com/kRLRzKF.png =88x)![](https://i.imgur.com/q1wvGs4.png =88x)![](https://i.imgur.com/TKQ43UD.png =88x)![](https://i.imgur.com/SIqA0rz.png =88x)![](https://i.imgur.com/6DSLyEG.png =88x)![](https://i.imgur.com/dlyo5ei.png =88x)![](https://i.imgur.com/LfAxpdR.png =88x)![](https://i.imgur.com/E5LrUS9.png =88x)
*Figure 13: Shift a biplane with the number of pixels from 1 to 8 (from left to right)*

To investigate whether PointCNN shows shift invariance on images, we shift the centre of CIFAR10 images by 1 to 8 pixels (*Figure 13*) and crop the images from 32×32 to 25×25. Then these "shifted" images are fed into PointCNN (trained on original CIFAR10 dataset) for testing. *Figure 14* shows how the prediction probability changes with the shifts.

![](https://i.imgur.com/M06LLLs.png =720x)
*Figure 14: Changes of predicted probability of the correct class using PointCNN when shifting the image (with a comparison with VGG16[^4])*

We check how often the network outputs the same classification, given the same image with two different shifts: $\mathbb{E}_{X, h1, w1, h2, w2}\{\arg\max P(\mathrm{Shift}_{h1, w1}(X)) = \arg\max P(\mathrm{Shift}_{h2, w2}(X))\}$, as introduced by Zhang et al.[^4] as the consistency score, where $h1, w1$ and $h2, w2$ denote two different shifts on the same image $X$. *Table 4* shows the result, where compared with VGG16, PointNet exhibits significantly lower consistency.

|                 | PointCNN |   VGG16  |
| :--------:      |:--------:|:--------:|
|Consistency Score| 63.67     | 88.52    |
*Table 4: Consistency scores (%) of PointCNN and VGG16[^4]*

## Extension: Image Segmentation

We also extend our experiments to image segmentation using PointNet and PointNet++, which to our best knowledge has not been discussed in any previous work. To train on 2D point clouds for semantic segmentation, we choose PASCAL VOC 2012, an image dataset designed for several recognition tasks (e.g. image classification, semantic segmentation, action classification, etc.) consisting of 20 classes.

![](https://i.imgur.com/ltiws5Q.jpg =400x)
*Figure 15: Two examples for semantic segmentation in PASCAL VOC 2012[^18]. Left: image. Right: corresponding segmentation label.*

Converting PASCAL VOC 2012 images to point clouds is similar to the method we use for CIFAR10. Considering the image size, 4096 pixels are randomly sampled from each image. *Table 4* shows the mean *IOU* of image segmentation result using PointNet and PointNet++. Notice that we have not achieved good results on this task, especially when compared to 2D CNNs listed in *Table 5*. During random sampling, a large number of local information is left out. Such a sampling method causes insufficient point density around the objects with semantics (e.g. the person on the motorbike in *Figure 15*), which affects the segmentation result.

| Network  | PASCAL VOC 2012 |
| :--------| :------: |
| PointNet | 9.36     |
|PointNet++| 14.54    |
|  FCN-8s[^10]\*  | 62.2     |
|  SANet[^11]\*   | 83.2     |
|DeepLabv3+[^12]\*| 89.0     |
*Table 5: Test mIOU (%) for semantic segmentation on PASCAL VOC 2012*

## Discussion

### Data Augmentation in PointCNN

> *To train the parameters in $\mathcal{X}$-Conv, it is evidently not beneficial to keep using the same set of neighbouring points, in the same order, for a specific representative point. To improve generalization, we propose to randomly sample and shuffle the input points, such that both the neighbouring point sets and order may differ from batch to batch[^3].*

Notice the above data augmentation technique used by the authors, we add another experiment, in which each input point cloud is shuffled before being fed into the network. However, due to the fact that the point cloud training data is produced by one-shot sampling from the images, instead of generated on the fly with each batch, the neighbouring point sets are fixed from batch to batch, thus the generalization effect is limited and eventually the performance is degraded from the reported.

### Evaluation of Shift Invariance

The consistency score proposed by Zhang et al.[^4] is strongly affected by the accuracy, i.e., when the accuracy is significantly high (e.g. 99%), the consistency, which measures how often the network outputs the same classification result given the same image with two different shifts, will be inevitably high as well. This dependent metric cannot depict the invariance under varied accuracy. Therefore, considering the significantly gaped accuracy between PointCNN and VGG16, the conclusion is difficult to make: which one is more shift invariant? By this we also consider that the use of nonorthogonal metrics in the paper[^4] on "Making the Convolutional Neural Networks Shift Invariant Again" is possibly inappropriate.

### Limitations of Our Method

Although extensive experiments and result analysis are provided, our method for image recognition using networks for point clouds still show some limitations:

* Datasets we mainly use in the project (i.e., MNIST, Fashion MNIST and CIFAR10) contain images with small sizes and a very limited number of categories.
* The random sampling method does not capture the shape information of objects in images in the best way and can cause an insufficient point density for the foreground object.
* We use a basic implementation of PointCNN, which can affect the performance and further comparisons.
* We do not extend image segmentation to PointCNN.

## Conclusion

In this project, we explored the possibility of using neural networks designed for point cloud analysis, namely PointNet, PointNet++ and PointCNN in image recognition, with an emphasis on image classification.

With extensive experiments, the results show that for images where shape information is prominent (e.g., MNIST and Fashion MNIST), all of the three networks are capable of classifying them. For images without explicit shape information (e.g., CIFAR10), only PointCNN can learn solid feature representation, while PointNet and PointNet++ fail completely. By extracting the objects from the backgrounds to facilitate shape information expression, both PointNet and PointNet++ shows significant accuracy improvement, which indicates shape information is superior to the RGB for PointNet and PointNet++.

From the comparison with the baselines, it is clear that the CNNs have a considerable advantage over all the three networks designed for point cloud analysis. Notice that PointCNN employs the CNN mechanism by performing the $\mathcal{X}$-Conv, to which we attribute the "success" of PointCNN on CIFAR10 dataset. For the extended segmentation task, the gap between 2D CNNs and PointNet/PointNet++ is even larger. Therefore we conclude that 2D CNNs are still dominant in image recognition.

For PointCNN, its consistency on shift invariance is investigated. We expected the consistency gain due to the random sampling which poses regularization to the network, while the result shows lower consistency compared to a 2D CNN, VGG16. Arguably the consistency metric is constraint by the accuracy. We also regard the low consistency as a consequence of the one-shot sampling mechanism which prevents the network to sufficiently learn the randomly distributed features from batch to batch.

## References

[^1]: Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 652-660).
[^2]: Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in neural information processing systems (pp. 5099-5108).
[^3]: Li, Y., Bu, R., Sun, M., Wu, W., Di, X., & Chen, B. (2018). Pointcnn: Convolution on x-transformed points. In Advances in neural information processing systems (pp. 820-830).
[^4]: Zhang, R. (2019). Making convolutional networks shift-invariant again. arXiv preprint arXiv:1904.11486.
[^5]: Rother, C., Kolmogorov, V., & Blake, A. (2004). "GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG), 23(3), 309-314.
[^6]: Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
[^7]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[^8]: Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013, February). Maxout networks. In International conference on machine learning (pp. 1319-1327).
[^9]: Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
[^10]: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.
[^11]: Zhong Z, Lin Z Q, Bidart R, et al. Squeeze-and-Attention Networks for Semantic Segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 13065-13074.
[^12]: Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.
[^13]: Ferguson, M., Ak, R., Lee, Y. T. T., & Law, K. H. (2017, December). Automatic localization of casting defects with convolutional neural networks. In 2017 IEEE international conference on big data (big data) (pp. 1726-1735). IEEE.
[^14]: OpenCV Documentation. Interactive Foreground Extraction using GrabCut Algorithm, URL: https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html. Accessed on June 30, 2020.
[^15]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
[^16]: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.
[^17]: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
[^18]: Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2012). The pascal visual object classes challenge 2012 (voc2012). Results.
