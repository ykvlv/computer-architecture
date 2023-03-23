# pylint:  disable=missing-module-docstring
# pylint:  disable=invalid-name
# pylint:  disable=consider-using-f-string
# pylint:  disable=missing-function-docstring
# pylint:  disable=missing-class-docstring

import json
from enum import Enum


class Opcode(str, Enum):
    '''
    Opcode
    '''
    LW = 'LW'  # A <- [B]
    SW = 'SW'  # [A] <- B
    LWI = 'LWI'  # A <- [IMM]
    SWI = 'SWI'  # [A] <- IMM

    JMP = 'JMP'  # unconditional transition
    # a,b,i
    BEQ = "BEQ"  # Branch if EQual (A == B)
    BNE = "BNE"  # Branch if Not Equal (A != B)
    BLT = "BLT"  # Branch if Less Than (A < B)
    BGT = "BGT"  # Branch if greater then (A > B)
    BNL = "BNL"  # Branch if Not Less than (A >= B)
    BNG = "BNG"  # Branch if less or equals then (A <= B)

    AND = 'AND'
    OR = 'OR'

    ANDI = 'ANDI'
    ORI = 'ORI'

    ADD = 'ADD'  # t,a,b
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    REM = "REM"

    ADDI = 'ADDI'  # t,a,i
    MULI = "MULI"
    SUBI = "SUBI"
    DIVI = "DIVI"
    REMI = "REMI"

    SEQ = "SEQ"  # Set if EQual (A == B)
    SNE = "SNE"  # Set if Not Equal (A != B)
    SLT = "SLT"  # Set if Less Than (A < B)
    SGT = "SGT"  # Set if greater then (A > B)
    SNL = "SNL"  # Set if Not Less than (A >= B)
    SNG = "SNG"  # Set if less or equals then (A <= B)

    SEQI = "SEQI"  # Set if EQual (A == B)
    SNEI = "SNEI"  # Set if Not Equal (A != B)
    SLTI = "SLTI"  # Set if Less Than (A < B)
    SGTI = "SGTI"  # Set if greater then (A > B)
    SNLI = "SNLI"  # Set if Not Less than (A >= B)
    SNGI = "SNGI"  # Set if less or equals then (A <= B)

    HALT = 'HALT'


ops_gr = {}
ops_gr["branch"] = {
    Opcode.JMP,
    Opcode.BEQ,
    Opcode.BNE,
    Opcode.BLT,
    Opcode.BNL,
    Opcode.BGT,
    Opcode.BNG
}
ops_gr["imm"] = {
    Opcode.ADDI,
    Opcode.SUBI,
    Opcode.MULI,
    Opcode.DIVI,
    Opcode.REMI,

    Opcode.SEQI,
    Opcode.SNEI,
    Opcode.SLTI,
    Opcode.SGTI,
    Opcode.SNLI,
    Opcode.SNGI,


    Opcode.LWI,
    Opcode.SWI,
}
ops_gr["arith"] = {

    Opcode.AND,
    Opcode.OR,

    Opcode.ADDI,
    Opcode.SUBI,
    Opcode.MULI,
    Opcode.DIVI,
    Opcode.REMI,

    Opcode.ADD,
    Opcode.SUB,
    Opcode.MUL,
    Opcode.DIV,
    Opcode.REM,

    Opcode.SEQ,
    Opcode.SNE,
    Opcode.SLT,
    Opcode.SGT,
    Opcode.SNL,
    Opcode.SNG,

    Opcode.SEQI,
    Opcode.SNEI,
    Opcode.SLTI,
    Opcode.SGTI,
    Opcode.SNLI,
    Opcode.SNGI,
}

STDIN, STDOUT = 6969, 9696


def write_code(filename: str, program: list):
    """Записать машинный код в файл."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps(program, indent=4))


def read_code(filename: str) -> list:
    """Прочесть машинный код из файла."""
    with open(filename, encoding="utf-8") as file:
        return json.loads(file.read())
