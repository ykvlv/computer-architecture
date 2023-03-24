# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import unittest

from src.translator import pre_process, tokenize, \
    build_ast, parse_expression, Translator
from src.isa import read_code


class TestTranslatatorHello(unittest.TestCase):

    def test_pre_process(self):
        raw = '''val hello[20] = 'Hello, world!';
                puts hello;'''
        post_processed = pre_process(raw)

        self.assertEqual(
            post_processed, "val hello[20] = 'Hello, world!'; puts hello;")

    def test_tokenize(self):
        tokens = tokenize(
            "val hello[20] = 'Hello, world!'; puts hello;"
        )
        print(tokens)
        self.assertEqual(
            tokens,
            ['val', 'hello', '@', '[', '20', ']', '=',
             "'Hello, world!'", ';', 'puts', 'hello']
        )

    def test_building_ast(self):
        tokens = ['val', 'hello', '@', '[', '20', ']',
                  '=', "'Hello, world!'", ';', 'puts', 'hello']
        ast = build_ast(tokens)

        benchmark = [['val', 'hello', '@', '[', '20', ']',
                      '=', "'Hello, world!'"], ['puts', 'hello']]
        for instr_idx, instr in enumerate(benchmark):
            self.assertEqual(ast[instr_idx], instr)
            self.assertEqual(ast[instr_idx], instr)

    def test_translate(self):
        ast = [['puts', "'Hello, World!'"]]
        translator = Translator()
        data_and_program, variables = translator.translate(ast)

        d_a_p = read_code("./tests/examples/correct/hello.json")
        self.assertDictEqual(variables, {})
        for instr_idx, instr in enumerate(d_a_p['program']):
            if isinstance(instr, dict):
                self.assertEqual(
                    data_and_program['program'][instr_idx]["opcode"],
                    instr["opcode"]
                )
                self.assertEqual(
                    data_and_program['program'][instr_idx]["args"],
                    instr["args"]
                )
            else:
                self.assertEqual(data_and_program['program'][instr_idx], instr)


class TestTranslatatorCat(unittest.TestCase):

    def test_pre_process(self):
        raw = '''val buffer[30];
                    gets buffer;
                    puts buffer;'''
        post_processed = pre_process(raw)

        self.assertEqual(
            post_processed, "val buffer[30]; gets buffer; puts buffer;")

    def test_tokenize(self):
        tokens = tokenize(
            "val buffer[30];  gets buffer; puts buffer;"
        )
        print(tokens)
        self.assertEqual(
            tokens,
            ['val', 'buffer', '@', '[', '30', ']', ';',
             'gets', 'buffer', ';', 'puts', 'buffer']
        )

    def test_building_ast(self):
        tokens = ['val', 'buffer', '@', '[', '30', ']',
                  ';', 'gets', 'buffer', ';', 'puts', 'buffer']
        ast = build_ast(tokens)

        benchmark = [['val', 'buffer', '@', '[', '30', ']'],
                     ['gets', 'buffer'], ['puts', 'buffer']]
        for instr_idx, instr in enumerate(benchmark):
            self.assertEqual(ast[instr_idx], instr)
            self.assertEqual(ast[instr_idx], instr)

    def test_translate(self):

        ast = [['val', 'buf'], ["get", "buf"],
               ["while", ["buf", "!=", "0"], [
                   ["put", "buf"],
                   ["get", "buf"]
               ]
                ]]

        translator = Translator()
        data_and_program, variables = translator.translate(ast)

        d_a_p = read_code("./tests/examples/correct/cat.json")
        self.assertDictEqual(variables, {'buf': 0})
        for instr_idx, instr in enumerate(d_a_p['program']):
            if isinstance(instr, dict):
                self.assertEqual(
                    data_and_program['program'][instr_idx]["opcode"],
                    instr["opcode"]
                )
                self.assertEqual(
                    data_and_program['program'][instr_idx]["args"],
                    instr["args"]
                )
            else:
                self.assertEqual(data_and_program['program'][instr_idx], instr)


class TestExpressionParser(unittest.TestCase):
    def test_parse_expression(self):
        self.assertEqual(parse_expression(["a", "*", "b"]), ("*", "a", "b"))
        self.assertEqual(parse_expression(
            "( a + b ) * b".split(" ")), ("*", ("+", "a", "b"), "b"))
        self.assertEqual(
            parse_expression("a + b - c / ( ( d - e ) * f )".split(" ")),
            ("-", ("+", "a", "b"), ("/", "c", ("*", ("-", "d", "e"), "f"))))
