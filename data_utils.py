from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os
import json
import rich


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore


import pandas as pd
import json
import ast

CODE_SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

def get_kodcode_questions(data_path="kodcode_data/kodcode-9k/train.parquet") -> Dataset:
    df = pd.read_parquet(data_path)
    df = df[["prompt", "reward_model", "extra_info", "solution"]]
    df["ground_truth"] = df["reward_model"].apply(lambda x: x["ground_truth"])
    df.drop(columns=["reward_model"], inplace=True)

    def to_chat_prompt(p: list[str]) -> list[dict[str, str]]:
        if p[-1]["role"] == "user":
            p[-1]["content"] = CODE_SYSTEM_PROMPT + "\n\n" + p[-1]["content"]
            
        return p

    df["prompt"] = df["prompt"].apply(to_chat_prompt)

    data = Dataset.from_pandas(df)
    return data

import io
import ast
import tokenize
from typing import Optional

def _strip_python_comments(code: str) -> str:
    sio = io.StringIO(code)
    out = []
    prev_end = (1, 0)
    for tok in tokenize.generate_tokens(sio.readline):
        ttype, tstr, start, end, _ = tok
        if ttype == tokenize.COMMENT:
            continue
        # 공백/개행 보존
        (sl, sc), (el, ec) = start, end
        if prev_end[0] < sl:
            out.append("\n" * (sl - prev_end[0]))
            prev_end = (sl, 0)
        if prev_end[1] < sc:
            out.append(" " * (sc - prev_end[1]))
        out.append(tstr)
        prev_end = (el, ec)
    return "".join(out)

class _DocstringStripper(ast.NodeTransformer):
    def _strip_in_body(self, body):
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            return body[1:]
        return body
    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)
        node.body = self._strip_in_body(node.body)
        return node
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body = self._strip_in_body(node.body)
        return node
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        node.body = self._strip_in_body(node.body)
        return node
    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        node.body = self._strip_in_body(node.body)
        return node

def strip_python_comments_and_docstrings(code: str) -> str:
    try:
        tree = ast.parse(code)
        tree = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(tree)
        try:
            code_no_docs = ast.unparse(tree)
        except Exception:
            code_no_docs = code
    except Exception:
        code_no_docs = code

    return _strip_python_comments(code_no_docs)