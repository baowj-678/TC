import sys
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from vocab.vocab import Vocab