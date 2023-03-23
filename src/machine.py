# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name

import logging
from collections import deque
import sys

from src.isa import Opcode, read_code, STDOUT, STDIN, ops_gr


class RegisterUnit:
    registers: list[int]
    rd: int
    rs1: int
    rs2: int

    def __init__(self, registers_count: int, stack_vertex: int, ) -> None:
        self.registers = [0] * registers_count
        self.registers[registers_count - 1] = stack_vertex
        self.rd = 0
        self.rs1 = 0
        self.rs2 = 0

    def latch_rd(self, number):
        self.rd = number

    def latch_srs1(self, number):
        self.rs1 = number

    def latch_rs2(self, number):
        self.rs2 = number

    def get_rs1_data(self):
        return self.registers[self.rs1]

    def get_rs2_data(self):
        return self.registers[self.rs2]

    def set_dest_data(self, data):
        if self.rd != 0:
            self.registers[self.rd] = int(data)


class ALU:
    output: int
    a: int
    b: int

    def __init__(self) -> None:
        self.output = 0
        self.a = 0
        self.b = 0

    def load(self, a, b):
        self.a = a
        self.a = b

    def compute(self, opcode) -> int:
        match opcode:
            case Opcode.AND | Opcode.ANDI:
                self.output = self.a and self.b
            case Opcode.OR | Opcode.ORI:
                self.output = self.a or self.b
            case Opcode.ADD | Opcode.ADDI:
                self.output = self.a + self.b
            case Opcode.SUB | Opcode.SUBI:
                self.output = self.a - self.b
            case Opcode.MUL | Opcode.MULI:
                self.output = self.a * self.b
            case Opcode.DIV | Opcode.DIVI:
                self.output = self.a // self.b
            case Opcode.REM | Opcode.REMI:
                self.output = self.a % self.b
            case Opcode.SEQ | Opcode.SEQI:
                self.output = self.a == self.b
            case Opcode.SNE | Opcode.SNEI:
                self.output = self.a != self.b
            case Opcode.SLT | Opcode.SLTI:
                self.output = self.a < self.b
            case Opcode.SNL | Opcode.SNLI:
                self.output = self.a >= self.b
            case Opcode.SGT | Opcode.SGTI:
                self.output = self.a > self.b
            case Opcode.SNG | Opcode.SNGI:
                self.output = self.a <= self.b
        self.output = int(self.output)
        return self.output


class BranchComparator:
    rs1: int
    rs2: int

    def __init__(self) -> None:
        self.rs1 = 0
        self.b = 0

    def load(self, rs1, rs2):
        self.rs1 = rs1
        self.rs2 = rs2

    def compare(self) -> tuple[bool, bool]:
        return self.rs1 == self.rs2, \
               self.rs1 < self.rs2


class IO:
    input_buffer: deque

    def __init__(self, input_tokens: list) -> None:
        self.input_buffer = deque(input_tokens)
        self.output_buffer: deque = deque()

    def eof(self):
        return not self.input_buffer

    def input(self):
        return self.input_buffer.popleft()

    def output(self, character):
        self.output_buffer.append(character)


class DataPath:
    program_memory: list
    data_memory: list
    program_counter: int
    data_address: int
    ru: RegisterUnit
    alu: ALU
    bc: BranchComparator
    io: IO

    imm_gen: int

    def __init__(
            self, data: list, program: list,
            stack_size: int, input_buffer: list):
        assert stack_size > 0, "the stack size must be greater than zero"
        self.data_address = 0
        self.program_counter = 0

        self.data_memory = data + [0] * stack_size
        self.program_memory = program
        self.io = IO([ord(token) for token in input_buffer])
        self.imm_gen = 0
        self.instr = {"opcode": Opcode("HALT"), "args": []}
        self.current_data = 0

        self.ru = RegisterUnit(5, stack_vertex=len(self.data_memory) - 1)
        self.alu = ALU()
        self.bc = BranchComparator()

    def select_instruction(self) -> Opcode:
        self.instr = self.program_memory[self.program_counter]
        self.program_counter += 1
        args = tuple(map(int, self.instr['args']))
        opcode = Opcode(self.instr["opcode"])
        if opcode is Opcode.JMP:
            self.imm_gen = args[0]
        elif opcode is Opcode.LWI:
            self.ru.rd = args[0]
            self.imm_gen = args[1]
        elif opcode is Opcode.LW:
            self.ru.rd = args[0]
            self.ru.rs1 = args[1]
        elif opcode is Opcode.SW:
            self.ru.rs1 = args[0]
            self.ru.rs2 = args[1]
        elif opcode is Opcode.SWI:
            self.ru.rs1 = args[0]
            self.imm_gen = args[1]
        elif opcode in ops_gr["branch"]:
            self.ru.rs1 = args[0]
            self.ru.rs2 = args[1]
            self.imm_gen = args[2]
        elif opcode in ops_gr['arithmetic']:
            self.ru.rd = args[0]
            self.ru.rs1 = args[1]
            if opcode in ops_gr["immediate"]:
                self.imm_gen = args[2]
            else:
                self.ru.rs2 = args[2]
        return opcode

    def latch_rs1_to_alu(self):
        self.alu.a = self.ru.get_rs1_data()

    def latch_rs2_to_alu(self):
        self.alu.b = self.ru.get_rs2_data()

    def latch_imm_to_alu(self):
        """Загружает непосредственное значение в ALU"""
        self.alu.b = self.imm_gen

    def compute(self, opcode: Opcode):
        self.alu.compute(opcode)

    def latch_address_to_memory(self):
        """Загружает целевой адрес в память"""

        if self.ru.get_rs1_data() == STDIN:
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.ru.get_rs1_data()
            self.current_data = self.data_memory[self.data_address]

    def store_data_to_memory_from_reg(self):
        """Загружает данные в память"""
        if self.ru.get_rs1_data() == STDOUT:
            self.io.output(
                chr(self.ru.get_rs2_data())
            )
        else:
            self.data_memory[self.ru.get_rs1_data()] = self.ru.get_rs2_data()

    def store_data_to_memory_from_imm(self):
        """Загружает данные в память"""
        if self.ru.get_rs1_data() == STDOUT:
            self.io.output(chr(self.imm_gen))
        else:
            self.data_memory[self.ru.get_rs1_data(
            )] = self.imm_gen

    def latch_address_to_memory_from_imm(self):
        if self.imm_gen == STDIN:
            if self.io.eof():
                raise EOFError
            self.current_data = self.io.input()
        else:
            self.data_address = self.imm_gen
            self.current_data = self.data_memory[self.data_address]

    def latch_reg_from_memory(self):
        """Значение из памяти перезаписывает регистр"""
        self.ru.set_dest_data(self.current_data)

    def latch_reg_from_alu(self):
        """АЛУ перезаписывает регистр"""
        self.ru.set_dest_data(self.alu.output)

    def latch_program_counter(self):
        """Перезаписывает значение PC из ImmGen"""
        self.program_counter = self.imm_gen

    def latch_regs_to_bc(self):
        """Загружает регистры в Branch Comparator."""
        self.bc.rs1, self.bc.rs2 = \
            self.ru.get_rs1_data(), self.ru.get_rs2_data()
        return self.bc.compare()


class ControlUnit:
    data_path: DataPath

    def __init__(self, data_path):
        self.data_path = data_path
        self._tick = 0

    def tick(self):
        """
        Счётчик тактов процессора.
        Вызывается при переходе на следующий такт.
        """
        self._tick += 1

    def current_tick(self):
        """Возвращает текущий такт."""
        return self._tick

    def decode_and_execute_instruction(self):
        opcode = self.data_path.select_instruction()
        self.tick()
        if opcode is Opcode.HALT:
            raise StopIteration()

        if opcode is Opcode.JMP:
            self.data_path.latch_program_counter()
        elif opcode in ops_gr["branch"]:
            equals, less = self.data_path.latch_regs_to_bc()
            self.tick()
            if any([
                opcode is Opcode.BEQ and equals,
                opcode is Opcode.BNE and not equals,
                opcode is Opcode.BLT and less,
                opcode is Opcode.BNL and not less,
                opcode is Opcode.BGT and not less and not equals,
                opcode is Opcode.BNG and (less or equals)
            ]):
                self.data_path.latch_program_counter()
        elif opcode is Opcode.LWI:
            self.data_path.latch_address_to_memory_from_imm()
            self.tick()
            self.data_path.latch_reg_from_memory()
        elif opcode is Opcode.LW:
            self.data_path.latch_address_to_memory()
            self.tick()
            self.data_path.latch_reg_from_memory()
        elif opcode is Opcode.SW:
            self.data_path.store_data_to_memory_from_reg()
        elif opcode is Opcode.SWI:
            self.data_path.store_data_to_memory_from_imm()
        else:
            if opcode in ops_gr["immediate"]:
                self.data_path.latch_imm_to_alu()
            else:
                self.data_path.latch_rs2_to_alu()
            self.data_path.latch_rs1_to_alu()
            self.data_path.compute(opcode)
            self.tick()
            self.data_path.latch_reg_from_alu()

        logging.debug('%s', self)
        self.tick()

    def __repr__(self):
        state = f"{{" \
                f"TICK: {self._tick}, " \
                f"PC: {self.data_path.program_counter}, " \
                f"ADDR: {self.data_path.data_address}}}"

        extra = ' '.join([str(reg) for reg in self.data_path.ru.registers])
        registers = f"{{[" \
                    f"rd: {self.data_path.ru.rd}, " \
                    f"rs1: {self.data_path.ru.rs1}, " \
                    f"rs2: {self.data_path.ru.rs2}, " \
                    f"imm: {self.data_path.imm_gen}]  " \
                    f"Regs {extra}}}"

        opcode = self.data_path.instr["opcode"]
        args = self.data_path.instr['args']
        action = f"{opcode} {' '.join([str(arg) for arg in args])}"
        alu = f"ALU [" \
              f"a:{self.data_path.alu.a} " \
              f"b:{self.data_path.alu.b} " \
              f"output:{self.data_path.alu.output}]"

        return f"{state:30} {registers:40} {alu:30} {action:30} "


def show_memory(data_memory):
    data_memory_state = ""

    for address, cell in enumerate(reversed(data_memory)):
        address = len(data_memory) - address - 1
        address_br = bin(address)[2:]
        address_br = (12 - len(address_br)) * "0" + address_br
        if isinstance(cell, (int, str)):
            cell = int(cell)
            # binary representation == br
            cell_br = bin(cell)[2:]
            cell_br = (32 - len(cell_br)) * "0" + cell_br

            data_memory_state += f"[{{{address:6}}}\
        [{address_br:12}]  -> [{cell_br:32}] = ({cell:12})\n"
        else:
            instr = f"{cell['opcode']:5}"
            instr += "   "
            instr += ', '.join(map(str, cell['args']))
            data_memory_state += f"[{{{address:6}}}\
        [{address_br:12}]  -> [{'?' * 32}] = {instr:12}\n"

    return data_memory_state


def simulation(data: list, program: list, input_tokens, stack_size, limit):
    """Запуск симуляции процессора.

    Длительность моделирования ограничена количеством выполненных инструкций.
    """
    logging.info("{ INPUT MESSAGE } [ `%s` ]", "".join(input_tokens))
    logging.info("{ INPUT TOKENS  } [ %s ]", ",".join(
        [str(ord(token)) for token in input_tokens]))

    data_path = DataPath(data, program, stack_size, input_tokens)
    control_unit = ControlUnit(data_path)
    instr_counter = 0

    try:
        while True:
            if not limit > instr_counter:
                print("too long execution, increase limit!")
                break
            control_unit.decode_and_execute_instruction()
            instr_counter += 1
    except EOFError:
        logging.warning('Input buffer is empty!')
    except StopIteration:
        pass

    return ''.join(data_path.io.output_buffer), \
           instr_counter, \
           control_unit.current_tick(), \
           show_memory(data_path.data_memory), \
           show_memory(data_path.program_memory)


def main(args):
    assert len(args) == 2, \
        "Wrong arguments: machine.py <code.json> <input>"
    code_file, input_file = args

    data_and_program = read_code(code_file)
    data = data_and_program["data"]
    program = data_and_program["program"]
    with open(input_file, encoding="utf-8") as file:
        input_text = file.read()
        input_token = []
        for char in input_text:
            input_token.append(char)
    input_token.append(chr(0))

    output, instr_counter, ticks, data_memory, program_memory = simulation(
        data, program,
        input_tokens=input_token,
        stack_size=25,
        limit=20000
    )
    logging.info("%s", f"Data memory map is\n{data_memory}")
    logging.info("%s", f"Program memory map is\n{program_memory}")

    print(f"Output is `{''.join(output)}`")
    print(f"instr_counter: {instr_counter} ticks: {ticks}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    main(sys.argv[1:])
