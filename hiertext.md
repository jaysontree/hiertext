# hiertext
A Pytorch implementation of Unified_Detector for scenetext detection

### purpose of this project
This project is for personal interests and selfstudy :clown_face:

The Unified Detctor model is originally from Google's Tensorflow project [Unified Detector](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector). Inspired by a torch implementation of [Max Deeplab](https://github.com/conradry/max-deeplab), this project is trying to build a similar model on Pytorch for further study on my spare time.

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

### remark
- the dims in original project config needs a huge number of Memory/FrameBuffer. eg. the mask instance output has shape [256, 384, 1024, 1024], for fp32 it takes `256 x 384 x 1024 x 1024 x 4 = 384 (GB)` along. since the best device I can access is a tesla V-100 with 32GB fb, I have to reduce the num of masks under 100 and limit the batchsize under 4.
- there are some code changes to reduce memory consumption. hopefully they will lead to same results. eg. use matmul to replace reducesum(expanddim elementwise multiply), use indexing for matched mask instead of multiply full matching matrix.
- the main net maxdeeplab used here may be quite different from the one in original project.