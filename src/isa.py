"""
Архитектура набора команд
"""

import json
from enum import Enum

STDIN, STDOUT = 6969, 9696


class Opcode(str, Enum):
    """Коды операций"""

    LW = 'LW'  # A <- [B]
    SW = 'SW'  # [A] <- B
    LWI = 'LWI'  # A <- [IMM]
    SWI = 'SWI'  # [A] <- IMM

    JMP = 'JMP'  # безусловный переход

    # a,b,i
    BEQ = "BEQ"  # Branch if equal (A == B)
    BNE = "BNE"  # Branch if not equal (A != B)
    BLT = "BLT"  # Branch if less than (A < B)
    BGT = "BGT"  # Branch if greater than (A > B)
    BNL = "BNL"  # Branch if not less than (A >= B)
    BNG = "BNG"  # Branch if less or equals then (A <= B)

    AND = 'AND'
    OR = 'OR'

    ANDI = 'ANDI'
    ORI = 'ORI'

    # t,a,b
    ADD = 'ADD'
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    REM = "REM"

    # t,a,i
    ADDI = 'ADDI'
    MULI = "MULI"
    SUBI = "SUBI"
    DIVI = "DIVI"
    REMI = "REMI"

    SEQ = "SEQ"  # Set if Equal (A == B)
    SNE = "SNE"  # Set if Not Equal (A != B)
    SLT = "SLT"  # Set if Less Than (A < B)
    SGT = "SGT"  # Set if greater than (A > B)
    SNL = "SNL"  # Set if Not Less than (A >= B)
    SNG = "SNG"  # Set if less or equals then (A <= B)

    SEQI = "SEQI"  # Set if Equal (A == B)
    SNEI = "SNEI"  # Set if Not Equal (A != B)
    SLTI = "SLTI"  # Set if Less Than (A < B)
    SGTI = "SGTI"  # Set if greater than (A > B)
    SNLI = "SNLI"  # Set if Not Less than (A >= B)
    SNGI = "SNGI"  # Set if less or equals then (A <= B)

    HALT = 'HALT'


ops_gr = {
    "branch": {
        Opcode.JMP,
        Opcode.BEQ,
        Opcode.BNE,
        Opcode.BLT,
        Opcode.BNL,
        Opcode.BGT,
        Opcode.BNG
    },
    "immediate": {
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
    },
    "arithmetic": {

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
}


def write_code(filename: str, program: dict[str, list]):
    """Записать машинный код в файл"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(json.dumps(program, indent=4))


def read_code(filename: str) -> dict[str, list]:
    """Прочесть машинный код из файла"""
    with open(filename, encoding="utf-8") as file:
        return json.loads(file.read())
