from .base_options import BaseOptions
import os


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--useRestore', type=bool, default=False, help='whether restore model or not')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--max_iteration', type=int, default=2001, help='iterations for batch_size samples')
        parser.add_argument('--save_info_freq', type=int, default=100, help='frequency of saving train result')
        parser.add_argument('--early_stop', default=5, type=int, help='early stopping')
        return parser
