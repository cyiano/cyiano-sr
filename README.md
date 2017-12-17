# cyiano-sr
自己用于管理版本而创建的project。

---
+ 本项目从许多tensorflow高star的项目中搬运代码拼接而成，然鹅还没打算展示，所以懒得写详细出处了，我就不信你们能找到这。
+ readme文档慢慢完善，反正现在代码自己还能看得懂。
+ 英文版的readme什么时候有？像我这种英语渣，当然是等中文的写完了慢慢翻译过来啦:)
---
## TODO
+ 自己下载好图像集，在主目录下新建一个文件夹```Images```，里面新建两个文件夹```train```和```test```，其中```train```里面是训练集和验证集（验证集随即从中选一定比例），```test```是测试集。
+ 在主目录下新建文件夹```checkpoint```和```result```，前者是存放训练模型参数和tensorboard视图的地方，后者是放之后测试的图像输出。
+ 如果需要感知损失，去下载一个VGG19的官方TF参数，文件名就别改了，放在```checkpoint```下。[下载地址](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)

## 训练
+ 输入```python dataset.py```，生成tfrecord文件；
+ 输入```python train.py```，想要从头训练的话可以输```python train.py --ckpt=False```

## 测试
+ 有两个测试文件

