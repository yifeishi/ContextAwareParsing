import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--obj_code_size', type=int, default=2054)
    parser.add_argument('--box_code_size', type=int, default=6)
    parser.add_argument('--feature_size', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=100)
    parser.add_argument('--max_sym_num', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--show_log_every', type=int, default=3)
    parser.add_argument('--save_log', action='store_true', default=False)
    parser.add_argument('--save_log_every', type=int, default=3)
    parser.add_argument('--save_snapshot', action='store_true', default=True)
    parser.add_argument('--save_snapshot_every', type=int, default=50)
    parser.add_argument('--no_plot', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=1)
    parser.add_argument('--lr_decay_every', type=float, default=1)

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/data/05/deepfusion/users/yifeis/sceneparsing/data/room_feature')
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args