from abc import ABC, abstractmethod
from collections import namedtuple
import random
import re
import logging
import json
from tqdm import tqdm
from math_utils import clean_expr_str
from grader import math_equal
from vllm import LLM,SamplingParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0
    


class LLMNode(Node):
    prompt = "Above is the previous solution steps, solve the problem in one next step:\n"

    def __init__(self, previous_state, model, tokenizer,prm_model, prm_tokenizer, parent=None, num_children=2):
        self.question = previous_state
        self.previous_state = previous_state
        self.parent = parent
        self.model = model
        self.tokenizer = tokenizer
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.state = self.make_a_move(previous_state)
        self.children = None
        self.num_children = num_children
        
    
    def find_children(self):
        if not self.children:
            self.children = [
                MyNode(self.state, self.model, self.tokenizer, self.prm_model, self.prm_tokenizer, self)
                for _ in range(self.num_children)
            ]
        return self.children

    def find_random_child(self):
        return MyNode(
            self.state,
            self.model,
            self.tokenizer,
            self.prm_model,
            self.prm_tokenizer,
            self
            )

    def extract_pred_answer(self, pred_str):
        pred = ""
        if "boxed" in pred_str:
            ans = pred_str.split("boxed")[-1]
            if len(ans) == 0:
                return ""
            elif ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
            pred = a
        return pred
        
    def is_terminal(self):
        # text = self.output
        # pred = self.extract_pred_answer(text)
        # return False if pred == "" else True
        return True if "boxed" in self.output else False


    def reward(self, ground_truth):
        assert self.is_terminal(), f"reward called on non-terminal node {self}"

        pred = self.extract_pred_answer(self.output)
        prediction = clean_expr_str(pred)
        ground_truth = clean_expr_str(ground_truth)
        
        logger.info(f"Solving path: \n{self.state}")
        if math_equal(prediction, ground_truth):
            return 1
        return -1

    def make_a_move(
        self,
        previous_state,
    ):
        input_text = self.prompt + previous_state + "\n\nStep "
        # inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        stop_tokens = ['Step']
        stop_token_id = [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]
        stop_token_id.append(self.tokenizer.eos_token_id)
        outputs = self.model.generate(
            [input_text],
            SamplingParams(
                max_tokens=168,
                # do_sample=True,
                top_p=0.9,
                top_k=100,
                temperature=1.3,
                # pad_token_id=self.tokenizer.eos_token_id,
                stop_token_ids=stop_token_id,
            )
        )
        self.output = outputs[0].outputs[0].text
        # print(self.output)
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=168,
        #     do_sample=True,
        #     top_p=0.9,
        #     top_k=100,
        #     temperature=1.3,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     eos_token_id=stop_token_id,
        # )
        # self.output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # len_prompt = len(input_text)
        if self.output.endswith("Step"):
            self.output = self.output[: -len("Step")]
        else:
            self.output = self.output[:]
        self.output = "Step " + self.output + '\n'
        
        pred = self.extract_pred_answer(self.output)
        # 提取final answer的部分
        if pred != "":
            index = self.output.find("boxed")
            self.output = self.output[:index + 7 + len(pred)]
            
        return self.previous_state + self.output


class QuestionNode(LLMNode):
    def make_a_move(self, previous_state):
        # for question node, the current state is the question
        self.output = previous_state
        return previous_state

class MyNode(LLMNode):
    prompt = """You are an expert in the field of mathematics reasoning.
!!!Important!!!
Output the string "Step" strictly before solving the problem.
If, at any point, the information gathered so far is sufficient to directly solve the problem, output the final solution in this format:
"Final answer:\\boxed{a number}"

Do not include any additional commentary or explanation.

Here are few shots:

"A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n\nStep 1: Determine the amount of blue fiber required. \n[blue_fiber] = 2 bolts  \n\nStep 2: Calculate the amount of white fiber required.\nThe problem states that the white fiber is half the amount of blue fiber.  \n[white_fiber] = [blue_fiber] / 2  \n[white_fiber] = 2 / 2 = 1 bolt  \n\nStep 3: Calculate the total amount of fiber required.  \n[total_fiber] = [blue_fiber] + [white_fiber]  \n[total_fiber] = 2 + 1 = 3 bolts  \n\nFinal answer:\\boxed{3} ",

"Janet sells ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\nStep 1: Calculate the total number of eggs laid daily.\n[total_eggs] = 16\n\nStep 2: Determine the total number of eggs consumed daily.\n[eggs_consumed] = [breakfast_eggs] + [muffin_eggs] = 3 + 4 = 7\n\nStep 3: Calculate the remaining eggs available for sale.\n[remaining_eggs] = [total_eggs] - [eggs_consumed] = 16 - 7 = 9\n\nStep 4: Compute the daily earnings from selling the remaining eggs.\n[daily_earnings] = [remaining_eggs] * [price_per_egg] = 9 * 2 = 18\n\nfinal answer:\\boxed{18}"

"""
