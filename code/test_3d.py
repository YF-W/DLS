import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BraTS2019', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DLS', help='exp_name')
parser.add_argument('--model', type=str, default='mine3d_v1', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--decoder', type=int, default=1, help='choose decoder')


FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

snapshot_path = "../model/{}_{}_{}_labeled/{}".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
test_save_path = "../model/{}_{}_{}_labeled/{}_predictions/".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
if FLAGS.dataset_name == "LA":
    FLAGS.labelnum = 8
elif FLAGS.dataset_name == "Pancreas_CT":
    FLAGS.labelnum = 6
elif FLAGS.dataset_name == "BraTS2019":
    FLAGS.labelnum = 25
num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [
        FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list
    ]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas/'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

elif FLAGS.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/BraTS2019/'
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)


def test_calculate_metric():

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    # net.cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model,
                                   FLAGS.decoder,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(112, 112, 80),
                                   stride_xy=18,
                                   stride_z=4,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model,
                                   FLAGS.decoder,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)

    elif FLAGS.dataset_name == "BraTS2019":
        avg_metric = test_all_case(FLAGS.model,
                                   FLAGS.decoder,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=64,
                                   stride_z=64,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
