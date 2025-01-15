import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from importlib import import_module

from segment_anything import sam_model_registry

from datasets.dataset_synapse import Synapse_train_dataset
from trainer.synapse_trainer import trainer_synapse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='npz_data', help='root dir for data') #'Abdomen/Training'
parser.add_argument('--output', type=str, default='output/sam/results')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--length', type=int, default=5, help='slice')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
# parser.add_argument('--lora_ckpt', type=str, default='output/sam/results/Synapse_512_pretrain_vit_b_epo200_length5_lr0.005_head3_half/best_epoch_0_0.9419.pth', help='Finetuned lora checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', type=bool, default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', type=bool, default=True, help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder_mask_decoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument("--multimask", type=bool, default=True, help="output multimask")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
parser.add_argument("--iter_point", type=int, default=5, help="point iterations")
args = parser.parse_args()

# '''synapse'''
# root_path = 'data/Synapse'
# dataset = 'Synapse'
# batch_size = 12
# img_size = 512
#
# '''refuge'''
# root_path = 'data/REFUGE-MultiRater'
# dataset = 'Refuge'
# batch_size = 1
# img_size = 1024

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name

    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_length' + str(args.length)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_head3_half'


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    '''dataset'''
    print("Preparing dataset.......")
    train_dataset = Synapse_train_dataset(base_dir=args.root_path, split='train_5', length=args.length, point_num=1)
    print("The length of train dataset is: {}".format(len(train_dataset)))
    batch_size = args.batch_size * args.n_gpu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    '''model'''
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    if args.lora_ckpt is not None:
        print("Loading pretrain parameters.......")
        net.load_lora_parameters(args.lora_ckpt)

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, train_loader, snapshot_path)

# dir = "Abdomen/Training"
# split = 'train'
# length = 6
# print(train_dataset[0]['image'].shape)
# print(train_dataset[0]['image'].unique())
# print(train_dataset[0]['label'].shape)
# print(train_dataset[0]['label'].unique())
# print(train_dataset[0]['point_coords'].shape)
# print(train_dataset[0]['point_labels'].shape)
# print(train_dataset[0]['organ'])
# print(train_dataset[0]['case_number'])
# print(train_dataset[0]['start_slice'])
# exit(-1)








