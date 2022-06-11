from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--result', type=str, default='../final_result', help='path to save prediction result')
        parser.add_argument('--savetest', type=str, default='../data/pre_process/testdata.hdf5',
                            help='path to test data')
        parser.add_argument('--mode', type=str, default='test')
        return parser
