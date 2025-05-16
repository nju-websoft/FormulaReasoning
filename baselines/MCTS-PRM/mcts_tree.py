from abc import ABC, abstractmethod
from collections import defaultdict
import math
import logging
from logging import getLogger
import random
from html import escape
from prm import get_step_reward

random.seed(1234)

logger = getLogger(__name__)
logger.setLevel("INFO")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(float)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.Reward = defaultdict(float)
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.root = None

    def choose(self, node, best):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        if best:
            scores = {n: self.Reward[n] for n in self.children[node]}
            max_score = max(scores.values())
            candidates = [n for n, s in scores.items() if s == max_score]
            return random.choice(candidates)
        else:
            scores = {n: self.Reward[n] for n in self.children[node]}
            min_score = min(scores.values())
            candidates = [n for n, s in scores.items() if s == min_score]
            return random.choice(candidates)

    def do_iteration(self, node):
        self.root = node
        "Make the tree one layer better. (Train for one iteration.)"
        logger.info("===== Start MCTS Iteration =====")
        logger.info("Step 1: Perform Selection")
        path = self._select(node)
        leaf = path[-1]
        
        logger.info(f"Selected leaf node type: {leaf.__class__.__name__}")
        logger.info("Step 2: Perform Expansion")
        if "boxed" not in leaf.output:
            self._expand(leaf)
        logger.info("Step 3: Perform Rollout")
        reward = self._rollout(leaf)
        logger.info(f"Rollout completed with Reward: {reward:.2f}")

        logger.info("Step 4: Perform Backpropagation")
        self._backpropagate(path, reward)

        logger.info("===== End MCTS Iteration =====")

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()
        
        # 调用Qwen2.5-PRM Model直接获取reward
        for child in self.children[node]:
            child_step_reward = get_step_reward(child)
            self.Reward[child] = child_step_reward[0][0]

    def _rollout(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        if self.Reward[node] == 0:
            step_reward = get_step_reward(node)
            self.Reward[node] = step_reward[0][0]
        return self.Reward[node]

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    

    def _visualize(self, prefix: str = ""):
        "Visualize the MCTS tree using Graphviz"
        from graphviz import Digraph

        dot = Digraph(comment='MCTS Tree')
        dot.attr('node', fontname='KaiTi', fontsize='10')
        dot.attr('edge', fontname='KaiTi', fontsize='8')

        def escape_text(text):
            return escape(str(text)) if text else " "

        def add_nodes_edges(node):
            node_id = str(id(node))
            if node == self.root:
                state = escape_text(node.state)
                label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="5" CELLPADDING="5">' \
                        f'<TR><TD ALIGN="LEFT" BGCOLOR="lightcoral"><B>R={self.Reward[node]:.2f}</B><BR/>N={self.N[node]}<BR/><I>{state}</I></TD></TR></TABLE>>'
                dot.node(node_id, label=label, shape='ellipse', width='1.2', height='0.8')
            else:
                output = escape_text(node.output)
                label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="5" CELLPADDING="5">' \
                        f'<TR><TD ALIGN="LEFT" BGCOLOR="lightblue"><B>R={self.Reward[node]:.2f}</B><BR/>N={self.N[node]}<BR/><I>{output}</I></TD></TR></TABLE>>'
                dot.node(node_id, label=label, shape='box', width='1.2', height='0.8')

            if node in self.children:
                for child in self.children[node]:
                    child_id = str(id(child))
                    add_nodes_edges(child)
                    dot.edge(node_id, child_id, color='gray', arrowhead='vee')
                

        add_nodes_edges(self.root)

        output_path = f'./visualization_mcts/{prefix}'
        dot.render(output_path, format='png', cleanup=True, view=False)
        logger.info(f"Tree visualization saved to {output_path}.png")
