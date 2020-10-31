import sys
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from MT_LSTM import *
from config import *