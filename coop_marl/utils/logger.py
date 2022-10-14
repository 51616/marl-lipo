import os
import logging

from torch.utils.tensorboard import SummaryWriter

logger = None

def get_logger(log_dir='tmp/', debug=False):
    global logger
    if logger is None:
        logger = Logger(log_dir, debug)
    return logger

def create_logger(log_dir='', debug=False):
    '''Create a global logger that logs INFO level messages to stdout and DEBUG ones to debug.log'''
    logger = logging.getLogger()
    log_formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt=None))
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    loc = 'debug.log'
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        loc = f'{log_dir}/debug.log'
    debug_handler = logging.FileHandler(loc, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def pblock(msg, msg_header=''):
    out = []
    out.append('='*60)
    out.append(msg_header)
    out.append(str(msg))
    out.append('='*60)
    return '\n'.join(out)

class Logger:
    """Combine logger and SummaryWriter into a single object"""
    def __init__(self, log_dir, debug):
        self.writer = SummaryWriter(log_dir)
        self.logger = create_logger(log_dir, debug)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif getattr(self.logger, attr, None):
            return getattr(self.logger, attr)
        else:
            return getattr(self.writer, attr)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        for k,v in tag_scalar_dict.items():
            self.writer.add_scalar(f'{main_tag}/{k}', v, global_step, walltime)

    def close(self):
        logging.shutdown()
        self.writer.close()

# taken from https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
# https://svn.blender.org/svnroot/bf-blender/trunk/blender/build_files/scons/tools/bcolors.py
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''