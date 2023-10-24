import os
from typing import Optional, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import state
from const import WORDLE_N, REWARD

import colorama
from colorama import Fore, Style

CUR_PATH = os.environ.get('PYTHONPATH', '.')
import os
dirname = os.path.dirname(__file__)
VALID_WORDS_PATH = f'{dirname}/data/wordle_words.txt'


def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]

def strToEncode(lines):
    encoding = []
    for line in lines:
        assert len(line.strip()) == WORDLE_N  # Must contain 5-letter words for now
        encoding.append(tuple(ord(char) - 65 for char in line.strip()))
    return encoding

class WordleEnvBase(gym.Env):
    """
    Actions:
        Can play any 5 letter word in vocabulary
        * 13k for full vocab
    State space is defined as:
        * 6 possibilities for turns (WORDLE_TURNS)
        * Each VALID_CHAR has a state of 0/1 for whether it's been guessed before
        * For each in VALID_CHARS [A-Z] can be in one of 3^WORDLE_N states: (No, Maybe, Yes)
        for full game, this is (3^5)^26
        Each state has 1 + 5*26 possibilities
    Reward:
        Reward is 10 for guessing the right word, -10 for not guessing the right word after 6 guesses.
    Starting State:
        Random goal word
        Initial state with turn 0, all chars Unvisited + Maybe
    """
    def __init__(self, words: List[str],
                 max_turns: int,
                 allowable_words: Optional[int] = None,
                 frequencies: Optional[List[float]]=None,
                 mask_based_state_updates: bool=False):
        assert all(len(w) == WORDLE_N for w in words), f'Not all words of length {WORDLE_N}, {words}'
        self.words = words
        self.encoded_words = strToEncode(words)
        self.max_turns = max_turns
        self.allowable_words = allowable_words
        self.mask_based_state_updates = mask_based_state_updates
        if not self.allowable_words:
            self.allowable_words = len(self.words)

        self.frequencies = None
        if frequencies:
            assert len(words) == len(frequencies), f'{len(words), len(frequencies)}'
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies)

        self.action_space = spaces.Discrete(len(self.words))
        self.observation_space = spaces.MultiDiscrete(state.get_nvec(self.max_turns))

        self.done = True
        self.goal_word: int = -1

        self.guesses = []
        self.state: state.WordleState = None
        self.state_updater = state.update
        if self.mask_based_state_updates:
            self.state_updater = state.update_mask

    def step(self, action: int):
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = self.state_updater(state=self.state,
                                        word=self.words[action],
                                        goal_word=self.words[self.goal_word])

        reward = 0
        if action == self.goal_word:
            self.done = True
            #reward = REWARD
            if state.remaining_steps(self.state) == self.max_turns-1:
                reward = 0 # No reward for guessing off the bat
            else:
                #reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns
                reward = REWARD
        elif state.remaining_steps(self.state) == 0:
            self.done = True
            reward = -REWARD

        # update game board and alphabet tracking
        board_row_idx = self.max_turns - state.remaining_steps(self.state) - 1
        encoded_guessed_word = self.encoded_words[action]
        for idx, char in enumerate(encoded_guessed_word):

            if self.encoded_words[self.goal_word][idx] == char:
                encoding = 2
                self.alphabet[char] = encoding
            elif char in self.encoded_words[self.goal_word]:
                encoding = 1
                if self.alphabet[char] == 0:
                    self.alphabet[char] = encoding
            else:
                encoding = 0

            self.board[board_row_idx, idx] = encoding
            # self.alphabet[char] = encoding

        # update previous guesses made
        self.guesses.append(self.encoded_words[action])

        return self.state.copy(), reward, self.done, {"goal_id": self.goal_word}

    def _get_obs(self):
        return {'board': self.board, 'alphabet': self.alphabet}

    def reset(self, seed: Optional[int] = None):
        self.state = state.new(self.max_turns)
        self.done = False
        self.goal_word = int(np.random.random()*self.allowable_words)
        self.board = np.negative(
            np.ones(shape=(self.max_turns, WORDLE_N), dtype=int))
        self.alphabet = np.zeros(shape=(26,), dtype=int)
        self.guesses = []

        return self.state.copy()

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORDLE_N):
                letter = chr(ord('A') + self.guesses[i][j])
                if self.board[i][j] == 0:
                    print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 1:
                    print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
                elif self.board[i][j] == 2:
                    print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            print()
        print()

        for i in range(len(self.alphabet)):
            letter = chr(ord('A') + i)
            if self.alphabet[i] == 0:
                print(Fore.BLACK + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 1:
                print(Fore.YELLOW + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == 2:
                print(Fore.GREEN + Style.BRIGHT + letter + " ", end='')
            elif self.alphabet[i] == -1:
                print(letter + " ", end='')
        print()
        print("HEY, GOAL WORD IS ", self.words[self.goal_word])
        print('###################################################')
        print()


    def set_goal_word(self, goal_word: str):
        self.goal_word = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int):
        self.goal_word = goal_id


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6)


class WordleEnv100OneAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=1, max_turns=6)


class WordleEnv100WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv100TwoAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=2, max_turns=6)


class WordleEnv100FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=100, max_turns=6)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6)


class WordleEnv1000WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv1000FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=1000, max_turns=6)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6)


class WordleEnvReal(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6)


class WordleEnvRealWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6,
                         mask_based_state_updates=True)