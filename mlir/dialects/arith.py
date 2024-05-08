""" Implementation of the Arith dialect. """

import inspect
import sys
from mlir.dialect import (Dialect, DialectOp, UnaryOperation, BinaryOperation,
                          is_op)
import mlir.astnodes as mast
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]

@dataclass
class ConstantOperation(DialectOp):
    value: Literal
    type: mast.Type
    _syntax_ = 'arith.constant {value.constant_literal} : {type.type}'

class AddiOperation(BinaryOperation): _opname_ = 'arith.addi'
class AddfOperation(BinaryOperation): _opname_ = 'arith.addf'

class MulfOperation(BinaryOperation): _opname_ = 'arith.mulf'
class MulIOperation(BinaryOperation): _opname_ = 'arith.muli'

arith = Dialect('arith', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])