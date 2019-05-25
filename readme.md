
face image illumination quality assessment  implements by pytorch 


Dependencies
------------------------------------------------------------------------------------
*python 3.6+
*pytorch 1.0.0+
*opencv2 (optimal)

Usage
------------------------------------------------------------------------------------
1.cloning the respository
git clone https://github.com/yangyuke001/FIIQA_pytorch

cd FIIQA_pytorch

2.Downloading the dataset

Face Image Illumination Quality Dataset 1.0

We have established a large-scale benchmark dataset, in which face images under various illumination patterns with associated illumination quality scores were constructed by making use of illumination transfer. Thus, we firstly collected an image set containing face images with various real-world illumination patterns, namely source illumination patterns set, and evaluated their illumination quality scores by subjective judgements. And after construction, this dataset is divided into three subsets for DCNN, the training set, the validation set and the testing set.
1. Source illumination patterns set (http://pan.baidu.com/s/1hrYayXI)

Unzip ZIP files, "illumination patterns.zip". In the "illumination patterns" folder, there are 200 images with various real-world illumination patterns, and for each image pattern, the associated illumination quality scores are given in the "patternsScores.mat", the sorted ranks, which are the class labels of those patterns, are given in the "patternsRank.mat".
2. Training Set (http://pan.baidu.com/s/1mhFBusg)

Unzip 7Z files, "trainingset.7z". In "train-faces" folder, there are 159159 images with various illumination patterns, and for each image the rank label of the associated illumination quality are given in the "train_standard.txt".
3. Validation Set (http://pan.baidu.com/s/1miMDkt6)

Unzip ZIP files, "validationset.zip". In "val-faces" folder, there are 30930 images with various illumination patterns, and for each image the rank label of the associated illumination quality are given in the "val_standard.txt".
4. Testing Set (http://pan.baidu.com/s/1nuXQjH3)



Unzip 7Z files, "testingset.7z". In "test-faces" folder, there are 34644 images with various illumination patterns, and for each image the rank label of the associated illumination quality are given in the "test_standard.txt".



