# hiertext
A Pytorch implementation of Unified_Detector for scenetext detection and layout analysis.

__i am still working on it on my spare time and any Advising/discussion is welcome ! 🤯:__

### update

2023-8-24 :calendar: the training on hiertext was not successful, model outputs are random strip-like shapes. using coco panoptic2017 dataset to train on the Maxdeeplab(from [Max Deeplab](https://github.com/conradry/max-deeplab)) was not successful. will working on it.

### purpose of this project
This project is for personal interests and selfstudy :clown_face:

The Unified Detctor model is originally from Google's Tensorflow project [Unified Detector](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector). This project is trying to build a similar model on Pytorch for further study on my spare time, inspired by a torch implementation of [Max Deeplab](https://github.com/conradry/max-deeplab). 

### TaskList
- [x] added a paragraph head to original maxdeeplab net
- [x] defined a paragraph grouping loss
- [x] modified loss computing code to support both raw and balanced style loss. 
- [x] modified original maxdeeplab code to solve some OOM issue
- [x] defined a simple Hiertext dataset for torch dataloader
- [ ] verrify the model
- [ ] train the model on Hiertext dataset

### reference
- [Unified Detector](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector)
- [Max Deeplab Unofficial](https://github.com/conradry/max-deeplab)
- [HierText dataset](https://github.com/google-research-datasets/hiertext)
- [Related ChineseBlog](https://blog.csdn.net/ckt20466498/article/details/127574980)

### remark
- the dims in original project config needs a huge number of Memory/FrameBuffer. eg. the mask instance output has shape [256, 384, 1024, 1024], for fp32 it takes `256 x 384 x 1024 x 1024 x 4 = 384 (GB)` along. since the best device I can access is a Nvidia GPU with 32G fb, I have to reduce the num of masks under 40 and limit the batchsize to 4.
- there are some code changes to reduce memory consumption. hopefully they will lead to same results. eg. use matmul to replace reducesum(expanddim elementwise multiply), use indexing for matched mask instead of multiply full matching matrix.
