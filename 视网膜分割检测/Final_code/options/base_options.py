# a few codes come from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import argparse


class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing.
    #It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # paras for path
        parser.add_argument('--dataroot', default='../data/raw_dataset', help='path to data')
        parser.add_argument('--datasave', default='../data/pre_process',
                            help='processed_data save path')
        parser.add_argument('--saveroot', default='../IoU', help='path to save Models')

        #paras for data
        parser.add_argument('--data_size', type=str, default='[640,400,400]',
                            help='input data size separated with comma')
        parser.add_argument('--crop_height', type=int, default=64, help='the height of the augmented pictures')
        parser.add_argument('--crop_width', type=int, default=64, help='the height of the augmented pictures')
        parser.add_argument('--stride_height', type=int, default=8, help='the stride of height of the test pictures')
        parser.add_argument('--stride_width', type=int, default=8, help='the stride of height of the test pictures')
        parser.add_argument('--input_nc', type=int, default=1, help='input channels')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

        #paras for device
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids')

        #paras for result
        parser.add_argument('--NUM_OF_CLASS', type=int, default=2, help='final class number for classification')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        print('')

    def parse(self):
        """Parse our options"""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
