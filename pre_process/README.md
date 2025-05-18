# 作用
接收mask_img图片,以及boxLabel标注，裁剪获取到框级别的图片，然后使用GlyphEncoder获取特征；
同时提取字体颜色的平均值；
裁剪canny获取到边缘图片，然后使用FontExtractor获取特征；
# 问题
1. 图片放缩后box有所变化，在放锁图片的同时放缩图片的box,修改posterDataLoader逻辑
2. 将图片的标注转换成yolo格式的图片，无惧放缩
