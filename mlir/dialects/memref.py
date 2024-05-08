""" Implementation of the Memref dialect. """

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
class DimOperation(DialectOp):
    operand: mast.SsaId
    index: mast.SsaId
    type: mast.Type
    _syntax_ = 'memref.dim {operand.ssa_id} , {index.ssa_id} : {type.type}'

# Memory Operations
@dataclass
class AllocOperation(DialectOp):
    args: mast.DimAndSymbolList
    type: mast.MemRefType
    _syntax_ = 'memref.alloc {args.dim_and_symbol_use_list} : {type.memref_type}'

@dataclass
class MemrefCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'memref.cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'