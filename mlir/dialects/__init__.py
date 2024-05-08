from .affine import affine as affine_dialect
from .standard import standard as std_dialect
from .scf import scf as scf_dialect
from .linalg import linalg
from .func import func as func_dialect
from .arith import arith as arith_dialect
from .memref import memref as memref_dialect

STANDARD_DIALECTS = [affine_dialect, std_dialect, scf_dialect, linalg, func_dialect, arith_dialect, memref_dialect]
