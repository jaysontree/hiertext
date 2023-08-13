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
- [] verrify the model
- [] train the model on Hiertext dataset

### reference
- [Unified Detector](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector)
- [Max Deeplab Unofficial](https://github.com/conradry/max-deeplab)
- [HierText dataset](https://github.com/google-research-datasets/hiertext)