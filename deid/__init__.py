# https://stackoverflow.com/a/40846742/2623170
# https://github.com/numpy/numpy/pull/432/commits/170ed4e3
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
