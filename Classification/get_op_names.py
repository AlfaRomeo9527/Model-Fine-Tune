
'''
获取预训练模型的按照网络结构顺序的结点名称。

Tensorflow 提供了主流的分类预训练模型。在训练自己的模型时，需要选择选练的层数以及获取最终层的结点名称。
由于预训练模型三.CKPT文件，通过。CKPT文集获取到的结点名词并未按照顺序进行排序。
但是，根据笔者的实验发现，先将预训练的CKPT文件转为。PB文件。通过PB文件获取结点的名称却是按照网络结构进行排序的。
因此，本文将以inception_resnet_v2为例，实现这个需求
'''
import tensorflow as tf
from tensorflow.python.platform import gfile
pre_train_model=r'./inception_resnet_v2_inf_graph.pb'

def get_op_names(model_path):
    with tf.gfile.FastGFile(model_path,"rb") as file:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def,name="")
        tensor_name_list=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name,'\n')


if __name__ == '__main__':
    get_op_names(pre_train_model)