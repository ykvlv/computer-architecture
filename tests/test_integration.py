# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import contextlib
import io
import os
import tempfile
import unittest

from src import machine
from src import translator


class TestCases(unittest.TestCase):

    def test_cat(self):
        # Создаём временную папку для скомпилированного файла
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/examples/sources/cat.alg"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/examples/example.in"

            # Собираем весь стандартный вывод в переменную stdout.
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                with self.assertLogs('', level='INFO') as logs:
                    translator.main([source, target])
                    machine.main([target, input_stream])
            self.assertEqual(
                stdout.getvalue(),
                'source LoC: 13 code instr: 21\n'
                'Output is `Good news, everyone!`\n'
                'instructions: 332 micro: 622 ticks: 1702\n'
            )
            self.assertEqual(
                logs.output[0],
                "INFO:root:{ INPUT MESSAGE } [ `Good news, everyone!\x00` ]"
            )
            self.assertEqual(
                logs.output[1],
                "INFO:root:{ INPUT TOKENS  } "
                "[ 71,111,111,100,32,110,101,119,115,44,"
                "32,101,118,101,114,121,111,110,101,33,0 ]"
            )

    def test_hello(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/examples/sources/hello.alg"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/examples/example.in"

            # Собираем весь стандартный вывод в переменную stdout.
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                translator.main([source, target])
                machine.main([target, input_stream])

            self.assertEqual(
                stdout.getvalue(),
                'source LoC: 3 code instr: 23\n'
                'Output is `Hello, World!`\n'
                'instructions: 111 micro: 139 ticks: 364\n'
            )

    def test_prop2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/examples/sources/prob2.alg"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/examples/example.in"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                # Собираем журнал событий по уровню INFO в переменную logs.
                # with self.assertLogs('', level='INFO') as logs:
                translator.main([source, target])
                machine.main([target, input_stream])

                self.assertEqual(
                    stdout.getvalue(),
                    'source LoC: 97 code instr: 107\n'
                    'Output is `4613732`\n'
                    'instructions: 1582 micro: 4189 ticks: 11566\n'
                )

    def test_prop5(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            source = "./tests/examples/sources/prob5.alg"
            target = os.path.join(tmpdirname, "machine_code.out")
            input_stream = "./tests/examples/example.in"

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                # Собираем журнал событий по уровню INFO в переменную logs.
                # with self.assertLogs('', level='INFO') as logs:
                translator.main([source, target])
                machine.main([target, input_stream])

                self.assertEqual(
                    stdout.getvalue(),
                    'source LoC: 131 code instr: 174\n'
                    'Output is `232792560`\n'
                    'instructions: 6500 micro: 17162 ticks: 47701\n'
                )
