# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=too-many-statements

import sys
from typing import NamedTuple, Any, Callable
from collections import deque
import re

from src.isa import Opcode, write_code, STDIN, STDOUT


def pre_process(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        # remove comments
        comment_idx = line.find("#")
        if comment_idx != -1:
            line = line[:comment_idx]
        # remove leading spaces
        line = line.strip()

        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\n", " ", text)

    return text


keywords = {"val", "while", "if"}
key_chars = {";", ","}
operators = {
    "=": 0,
    "==": 1,
    "!=": 1,
    "<": 1,
    ">": 1,
    ">=": 1,
    "<=": 1,
    "&": 1,
    "|": 1,
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
    "@": 3
}
parenthesis = {
    "[": 4,
    "]": -4
}
brackets = {
    "(": 4,
    ")": -4
}

braces = {
    "{": 4,
    "}": -4
}

breakers = set().union(operators) \
    .union(parenthesis).union(brackets) \
    .union(braces).union(key_chars)
breakers.add(";")

breakers_by_length: dict = {0: set()}


def get_breaker_by_length(breaker_length):
    return set(
        map(
            lambda breaker: breaker[:breaker_length], filter(
                lambda breaker: len(breaker) >= length,
                breakers
            )
        )
    )


for length in range(1, max(map(len, breakers)) + 1):
    breakers_by_length[length] = get_breaker_by_length(length)
breakers_by_length[max(list(breakers_by_length)) + 1] = set()


def tokenize(processed_text):
    # parts will contain part of source text divided by "'"
    # every uneven part is a string literal and that don't need to tokenize
    # thus it's a token itself
    parts = deque()
    part_start_idx = 0
    for char_idx, char in enumerate(processed_text):
        if char == "'" \
                and char_idx > 0 \
                and processed_text[char_idx - 1] != "\\":
            parts.append(processed_text[part_start_idx:char_idx])
            part_start_idx = char_idx + 1
    parts.append(processed_text[part_start_idx:])

    tokens = deque()

    for part_idx, part in enumerate(parts):
        if part_idx % 2 != 0:
            tokens.append(f"'{part}'")
            continue
        # such difficulty only for one sort of cases: =,!=,==
        n_part = bytearray()
        keyword = bytearray()
        for char in part:
            if keyword:
                if f"{keyword.decode('utf-8')}{char}" in \
                        breakers_by_length[len(keyword) + 1]:
                    keyword.append(ord(char))
                else:
                    if keyword.decode("utf-8") in breakers:
                        n_part.append(ord(" "))
                        n_part.extend(keyword)
                        n_part.append(ord(" "))
                        keyword.clear()
                    else:
                        n_part.extend(keyword)
                        keyword.clear()
                    n_part.append(ord(char))
            elif char in breakers_by_length[1]:
                keyword.append(ord(char))
            else:
                n_part.append(ord(char))

        part = n_part.decode("utf-8")

        part = part.replace("[", " @ [")
        part = part.replace("]", " ] ")

        part = re.sub(" +", " ", part)

        tokens.extend(filter(lambda token: token, part.split(" ")))
    return list(tokens)


class ASTParserUnit:
    """
    Промежуточное представление
    """

    def __init__(self) -> None:
        self.terms: deque = deque()
        self.nesting_level: deque = deque()

    def get_current_term(self) -> deque:
        max_level = len(self.nesting_level)
        if max_level == 0:
            return self.terms
        term = self.terms[self.nesting_level[0]]
        for i in range(1, max_level):
            if isinstance(term[self.nesting_level[i]], list):
                term = term[self.nesting_level[i]]
            else:
                return term
        return term

    def drive_in(self):
        term = self.get_current_term()
        term.append([])
        self.nesting_level.append(len(term) - 1)

    def add_token(self, token):
        self.get_current_term().append(token)

    def drive_out(self):
        term = self.get_current_term()
        if term and not term[-1]:
            term.pop()
        self.nesting_level.pop()

    def export_terms(self):
        if self.terms and not self.terms[-1]:
            self.terms.pop()
        return list(self.terms)


def build_ast(tokens: list[str]):
    # memorized all variables
    parse_unit = ASTParserUnit()
    parse_unit.drive_in()
    for token in tokens:
        if token in ['if', 'while']:
            parse_unit.add_token(token)
        elif token == '(':
            parse_unit.drive_in()
        elif token == '{':
            parse_unit.drive_in()
            parse_unit.drive_in()
        elif token == ')':
            parse_unit.drive_out()
        elif token == '}':
            parse_unit.drive_out()
            parse_unit.drive_out()
            parse_unit.drive_out()
            parse_unit.drive_in()
        elif token == ';':
            parse_unit.drive_out()
            parse_unit.drive_in()
        else:
            parse_unit.add_token(token)

    return parse_unit.export_terms()


class Expression(NamedTuple):
    left: str | Any
    right: str | Any
    operation: Callable[[str, str], str]


PRIOR = 0
OP = 1
INDEX = 1


def last_min_op(exprs):
    last = 0
    for expr_idx, expr in enumerate(exprs):
        if expr[PRIOR] <= exprs[last][PRIOR]:
            last = expr_idx
    return last


def parse_expression(tokens):
    # linearization
    while any(map(lambda t: isinstance(t, list), tokens)):
        linearized_tokens = deque()
        for token in tokens:
            if isinstance(token, list):
                linearized_tokens.append("(")
                linearized_tokens.extend(token)
                linearized_tokens.append(")")
            else:
                linearized_tokens.append(token)
        tokens = linearized_tokens
    operation_priority = 0
    expr = deque()
    ops = deque()
    for token_idx, token in enumerate(tokens):
        if token in brackets:
            operation_priority += brackets[token]
        elif token in parenthesis:
            operation_priority += parenthesis[token]
        elif token in operators:
            priority = operation_priority + operators[token]
            expr.append((priority, token))
        else:
            expr.append(token)
    for token_idx, token in enumerate(expr):
        if isinstance(token, tuple):
            ops.append((token[0], token_idx))
    expr = list(expr)

    def resolve(ops: list[tuple]):
        op_ptr = last_min_op(ops)
        if op_ptr == len(ops) - 1:
            right = expr[ops[op_ptr][INDEX] + 1]
        else:
            right = resolve(ops[op_ptr + 1:])
        if op_ptr == 0:
            left = expr[ops[op_ptr][INDEX] - 1]
        else:
            left = resolve(ops[:op_ptr])

        return expr[ops[op_ptr][INDEX]][OP], left, right

    return resolve(list(ops)) if len(ops) else tokens[0]


class Translator:
    ARGS = "args"
    OPCODE = "opcode"
    FISH = "undefined_address"

    def __init__(self) -> None:
        self.program: deque = deque()

        self.vars: dict = {}
        self.memory: deque = deque()
        self.data_address = 0
        [self.X0, self.X1, self.X2, self.X3, self.SP] = \
            ["0", "1", "2", "3", "4"]
        self.op = {}
        self.pc = 0
        self.unresolved_addresses: deque = deque()
        for opcode in list(Opcode):
            self.op[opcode.name] = self.generate_commands(opcode)

    def generate_commands(self, opcode):

        def insert_command(rt=None, rla=None, rrb=None):
            args = []
            for arg in rt, rla, rrb:
                if arg is not None:
                    args.append(arg)
            self.program.append(
                {
                    self.OPCODE: opcode,
                    self.ARGS: args
                }
            )
            self.pc += 1

        return insert_command

    def append_unresolved_address(self):
        self.unresolved_addresses.append(self.pc - 1)

    def resolve_address(self):
        address = self.unresolved_addresses.pop()
        self.program[address][self.ARGS][2] = self.pc

    def solve(self, expr):
        if isinstance(expr, tuple):
            operation, left, right = expr
            self.solve(left)
            self.solve(right)
            if operation == '@':
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X2, self.SP)
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X1, self.SP)
                self.op["ADD"](self.X3, self.X1, self.X2)
                self.op["LW"](self.X1, self.X3)
                self.op["SW"](self.SP, self.X1)
                self.op["SUBI"](self.SP, self.SP, 1)

            else:
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X2, self.SP)
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X1, self.SP)

                match operation:
                    case '==':
                        self.op["SEQ"](self.X3, self.X1, self.X2)
                    case '!=':
                        self.op["SNE"](self.X3, self.X1, self.X2)
                    case '>':
                        self.op["SGT"](self.X3, self.X1, self.X2)
                    case '<':
                        self.op["SLT"](self.X3, self.X1, self.X2)
                    case '>=':
                        self.op["SNL"](self.X3, self.X1, self.X2)
                    case '<=':
                        self.op["SNG"](self.X3, self.X1, self.X2)
                    case '&':
                        self.op["AND"](self.X3, self.X1, self.X2)
                    case '|':
                        self.op["OR"](self.X3, self.X1, self.X2)
                    case '+':
                        self.op["ADD"](self.X3, self.X1, self.X2)
                    case '-':
                        self.op["SUB"](self.X3, self.X1, self.X2)
                    case '*':
                        self.op["MUL"](self.X3, self.X1, self.X2)
                    case '/':
                        self.op["DIV"](self.X3, self.X1, self.X2)
                    case '%':
                        self.op["REM"](self.X3, self.X1, self.X2)

                self.op["SW"](self.SP, self.X3)
                self.op["SUBI"](self.SP, self.SP, 1)
        else:
            if expr.isdigit():
                self.op["SWI"](self.SP, int(expr))
                self.op["SUBI"](self.SP, self.SP, 1)
            elif expr in self.vars:
                self.op["LWI"](self.X1, self.vars[expr])
                self.op["SW"](self.SP, self.X1)
                self.op["SUBI"](self.SP, self.SP, 1)

    def addr(self, addr):
        if isinstance(addr, tuple):
            _, left, right = addr
            self.solve(left)
            self.solve(right)
            self.op["ADDI"](self.SP, self.SP, 1)
            self.op["LW"](self.X2, self.SP)
            self.op["ADDI"](self.SP, self.SP, 1)
            self.op["LW"](self.X1, self.SP)
            self.op["ADD"](self.X3, self.X1, self.X2)
            self.op["SW"](self.SP, self.X3)
            self.op["SUBI"](self.SP, self.SP, 1)
        else:
            self.op["ADDI"](self.X1, self.X0, self.vars[addr])
            self.op["SW"](self.SP, self.X1)
            self.op["SUBI"](self.SP, self.SP, 1)

    def allocate(self, term: list):
        if term[0] == 'val':
            variable = term[1]
            if len(term) > 5 and (term[3], term[5]) == ('[', ']'):
                self.vars[variable] = self.data_address
                self.data_address += 1
                self.memory.append(self.data_address)

                self.data_address += int(term[4])
                if len(term) > 6:

                    if isinstance(term[7], list):
                        inits = list(filter(lambda val: val != ',', term[7]))
                        self.memory.extend(map(int, inits))
                        self.memory.extend([0] * (int(term[4]) - len(inits)))
                    elif (term[7][0], term[7][-1]) == ("'", "'"):
                        inits = list(map(ord, term[7][1:-1]))
                        self.memory.extend(map(int, inits))
                        self.memory.extend([0] * (int(term[4]) - len(inits)))

                else:
                    self.memory.extend([0] * int(term[4]))
            else:
                self.vars[variable] = self.data_address
                self.data_address += 1
                if len(term) > 2 and term[2] == '=':
                    value = term[3]
                    if value.isdigit():
                        self.memory.append(int(value))
                    else:
                        self.memory.append(self.memory[self.vars[variable]])
                else:
                    self.memory.append(0)

    def _translate(self, term: list):
        if not term:
            return
        match term[0]:
            case 'val':
                self.allocate(term)
            case "if" | "while":
                keyword = term[0]
                condition = term[1]
                body = term[2]
                condition_start = self.pc
                self.solve(parse_expression(condition))
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X1, self.SP)
                self.op["BEQ"](self.X1, self.X0, self.FISH)
                self.append_unresolved_address()
                # block init

                for _term_ in body:
                    self._translate(_term_)

                # end blocks
                if keyword == 'while':
                    self.op["JMP"](condition_start)
                self.resolve_address()
            case 'get':
                m_value = parse_expression(term[1:])
                self.addr(m_value)
                self.op["ADDI"](self.SP, self.SP, 1)
                self.op["LW"](self.X1, self.SP)
                self.op["LWI"](self.X3, STDIN)
                self.op["SW"](self.X1, self.X3)
            case 'put':
                if term[1].isdigit():
                    self.op["ADDI"](self.X1, self.X0, term[1])
                else:
                    vvalue = parse_expression(term[1:])
                    self.solve(vvalue)
                    self.op["ADDI"](self.SP, self.SP, 1)
                    self.op["LW"](self.X1, self.SP)
                self.op["ADDI"](self.X2, self.X0, STDOUT)
                # self.op["ADDI"](self.X1, self.X1, ord('0'))
                self.op["SW"](self.X2, self.X1)

            case 'gets':
                self.op["LWI"](self.X2, self.vars[term[1]])
                self.op["LWI"](self.X1, STDIN)
                self.op["BEQ"](self.X1, self.X0, self.FISH)
                self.append_unresolved_address()
                self.op["SW"](self.X2, self.X1)
                self.op["ADDI"](self.X2, self.X2, 1)
                self.op["JMP"](self.pc - 4)
                self.resolve_address()
            case 'puts':
                if (term[1][0], term[1][-1]) == ("'", "'"):
                    self.op["SW"](self.SP, self.X0)
                    self.op["ADD"](self.X3, self.SP, self.X0)
                    self.op["SUBI"](self.SP, self.SP, 1)
                    for character in (term[1])[1:-1]:
                        self.op["ADDI"](self.X1, self.X0, ord(character))
                        self.op["SW"](self.SP, self.X1)
                        self.op["SUBI"](self.SP, self.SP, 1)
                    # write string
                    self.op["ADD"](self.SP, self.X3, self.X0)

                    self.op["SUBI"](self.SP, self.SP, 1)
                    self.op["LW"](self.X1, self.SP)
                    self.op["SW"](self.SP, self.X0)
                    self.op["BEQ"](self.X1, self.X0, self.FISH)
                    self.append_unresolved_address()

                    self.op["ADDI"](self.X2, self.X0, STDOUT)
                    self.op["SW"](self.X2, self.X1)

                    self.op["JMP"](self.pc - 6)

                    self.resolve_address()

                    self.op["ADD"](self.SP, self.X3, self.X0)
                else:
                    self.op["LWI"](
                        self.X2, self.vars[term[1]])
                    self.op["LW"](self.X1, self.X2)
                    self.op["BEQ"](self.X1, self.X0, self.FISH)
                    self.append_unresolved_address()

                    self.op["ADDI"](self.X3, self.X0, STDOUT)
                    self.op["SW"](self.X3, self.X1)
                    self.op["ADDI"](self.X2, self.X2, 1)
                    self.op["JMP"](self.pc - 5)
                    self.resolve_address()
            case _:
                operation = term.index('=')
                m_value = parse_expression(term[:operation])
                expression = parse_expression(term[operation + 1:])
                self.addr(m_value)
                if (expression[0], expression[-1]) == ("'", "'"):
                    self.op["ADDI"](self.SP, self.SP, 1)
                    self.op["LW"](self.X2, self.SP)
                    self.op["LW"](self.X3, self.X2)
                    for char in expression[1:-1]:
                        self.op["SWI"](self.X3, ord(char))
                        self.op["ADDI"](self.X3, self.X3, 1)
                else:
                    self.solve(expression)
                    self.op["ADDI"](self.SP, self.SP, 1)
                    self.op["LW"](self.X2, self.SP)
                    self.op["ADDI"](self.SP, self.SP, 1)
                    self.op["LW"](self.X1, self.SP)
                    self.op["SW"](self.X1, self.X2)

    def translate(self, terms):
        self.unresolved_addresses = deque()
        self.program = deque()
        self.pc = 0

        for term in terms:
            self._translate(term)
        self.op["HALT"]()

        program = {
            "data": list(self.memory),
            "program": list(self.program)
        }

        return program, self.vars


def main(args):
    assert len(args) == 2, \
        "Wrong arguments: translator.py <input_file> <target_file>"

    source, target = args

    with open(source, "rt", encoding="utf-8") as f:
        source = f.read()

    text = pre_process(source)
    tokens = tokenize(text)
    ast = build_ast(tokens)
    translator = Translator()
    program, _ = translator.translate(ast)
    print("source LoC:", len(source.split()), "code instr:",
          len(program))

    write_code(target, program)


if __name__ == '__main__':
    main(sys.argv[1:])
