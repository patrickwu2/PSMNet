import argparse


def get_args():

    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument(
        '--maxdisp',
        type=int,
        default=192,
        help='max diparity'
    )
    parser.add_argument(
        '--datadir',
        # default="/work/kaikai4n/resized_stereo/",
        default="/tmp2/tsunghan/disparity_data/scannet_disparity_data/",
        help='data directory'
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=4,
        help='gpu number'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch size'
    )
    parser.add_argument(
        '--validate-batch-size',
        type=int,
        default=4,
        help='batch size'
    )
    parser.add_argument(
        '--log-per-step',
        type=int,
        default=1,
        help='log per step'
    )
    parser.add_argument(
        '--save-per-epoch',
        type=int,
        default=1,
        help='save model per epoch'
    )
    parser.add_argument(
        '--experiment-dir',
        default='experiment',
        help='directory where save model, log, images'
    )
    parser.add_argument(
        '--model-path',
        default=None,
        help='path of model to load'
    )
    # parser.add_argument(
    #   '--start-step',
    #   type=int,
    #   default=0,
    #   help='number of steps at starting'
    # )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=300,
        help='number of training epochs'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='num workers in loading data'
    )
    args = parser.parse_args()
    return args
