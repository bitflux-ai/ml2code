import sys
import os

tgbase = os.path.join(os.path.dirname(__file__), '..', 'tinygrad')
sys.path.append(tgbase)
tgbase = os.path.join(os.path.dirname(__file__), '..', 'tinygrad/tinygrad')
sys.path.append(tgbase)
tgbase = os.path.join(os.path.dirname(__file__), '..', 'onnxhack')
sys.path.append(tgbase)

from ml2code.cli import main