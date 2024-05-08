""" Implementation of the Arith dialect. """

import inspect
import sys
from mlir.dialect import (Dialect, DialectOp, UnaryOperation, BinaryOperation,
                          is_op)
import mlir.astnodes as mast
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class ConstantOperation(DialectOp):
    value: Literal
    type: mast.Type
    _syntax_ = 'arith.constant {value.constant_literal} : {type.type}'

class AddiOperation(BinaryOperation): _opname_ = 'arith.addi'
class AddfOperation(BinaryOperation): _opname_ = 'arith.addf'

class MulfOperation(BinaryOperation): _opname_ = 'arith.mulf'
class MulIOperation(BinaryOperation): _opname_ = 'arith.muli'