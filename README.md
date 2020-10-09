# KB-REF_dataset (Knowledge Based Referring Expression)
## Description
KB-REF dataset is a referring expression comprehension dataset. Different with other referring expression dataset, it requires that each referring expression must use at least one external knowledge (the information can not be got from the image). There are 31, 284 expressions with 9, 925 images in training set, 4, 000 expressions with 2, 290 images in validation set, and 8, 000 expressions with 4, 702 images in test set. Also the dataset contains a number of object categories.
## Download
The dataset can be downloaded from [BaiduYun Drive (code: hijs)](https://pan.baidu.com/s/1peAeva32dc5ZjK12-y4omw). The images of KB-REF are come from [VisualGenome](http://visualgenome.org/). It contains several files:
* expression.json: The main part of our dataset. It is a dictionary file： the key in the file is composed of image id (before the '\_') and object id (after the '\_'), the value is composed of the referring expression (first) and the corresponding fact (second).
* candidate.json: It is the ground truth objects for each image. For each image, we choose 10 ground truth objects as the candidate bounding box when the model is reasoning on the dataset. The key is the image id, and the value is the object id in the image.
* image.json: It contains the width and height of each image. The key is image id, the value is width (first), height (second).
* objects.json: It contains the specific information for each object instance. It is a two-tier dictionary file. The key for first tier is the image id. The key for second tier is the object instance id. The value contains: the object category, the object name, the x of the top left corner, the y of the top left corner, the width and the height of the bounding box.
* train.json, val.json, test.json: We split dataset according to the image. These files descripe which pictures are for train, val, and test.
* Vocabualry.json: The vocabulary file.
* Wikipedia.json, ConceptNet.json, WebChild.json: The knowledge we collect. The key is the object category and the value is the corresponding facts.

If you have any question about this dataset, please email to 1226726279@qq.com directly. And I will response you as as soon as possible.
