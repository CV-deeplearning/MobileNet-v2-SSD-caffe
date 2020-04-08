# MobileNet-v2-SSD-caffe 
# caffe SSD训练自己的数据

## 一 生成lmdb
* home目录下创建文件夹 data/VOCdevkit。
* cd ~/data/VOCdevkit 在该目录下创建MyDataSet, 在MyDataSet下创建文件夹Annotation和JPEGImages分别存放图片文件和标注文件。
* 将get_list.py 放在MyDataSet同级目录下，运行get_list.py 生成test.txt test_name_size.txt trainval.txt三个文件。
*
* 进入~/caffe_ssd/data/MyDataSet, 把上面的三个文件拷贝到该目录下。
* 修改labelmap_voc.prototxt, 运行create_data.sh 生成lmdb
* lmdb存放地址：~/data/VOCdevkit/MyDataSet/lmdb
*             ~/caffe_ssd/example/MyDataSet(存放数据链接，可删除)

## 二 训练模型
* 进入~/work/project/MobileNet-SSD
* 运行 sh gen_model.sh 4 生成对应的模型（4是检测的类别数，根据需要填写,背景也算1类）。
* 进入example目录， 修改train.prototxt和test.prototxt.(若无特殊要求，只修改路径即可)
* cd ../
* sh train.sh
* 模型训练中。。。

## 三 测试模型
* test_draw.py

* 整个过程走完，遇到错误，随手调调即可。
