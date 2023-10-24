import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pkg_resources
import random
from typing import Optional, List
import colorama
from colorama import Fore
from colorama import Style

from gym_wordle.exceptions import InvalidWordException

colorama.init(autoreset=True)

# global game variables
GAME_LENGTH = 6
WORD_LENGTH = 5

# load words and then encode
filename = 'data/wordle_words.txt'

def encodeToStr(encoding):
    string = ""
    for enc in encoding:
        string += chr(ord('A') + enc)
    return string

def strToEncode(lines):
    encoding = []
    for line in lines:
        assert len(line.strip()) == 5  # Must contain 5-letter words for now
        encoding.append(tuple(ord(char) - 65 for char in line.strip()))
    return encoding


def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(filename, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines # returns array of ALL the words in the .txt file (since limit = None by default)
        else:
            return lines[:limit] # returns an array of some words e.g. ['CIGAR', 'REBUT', 'SISSY']

TEXT_WORDS = _load_words()
WORDS = strToEncode(TEXT_WORDS)

class WordleEnv(gym.Env):
    """
    Simple Wordle Environment

    Wordle is a guessing game where the player has 6 guesses to guess the
    5 letter hidden word. After each guess, the player gets feedback on the
    board regarding the word guessed. For each character in the guessed word:
        * if the character is not in the hidden word, the character is
          grayed out (encoded as 0 in the environment)
        * if the character is in the hidden word but not in correct
          location, the character is yellowed out (encoded as 1 in the
          environment)
        * if the character is in the hidden word and in the correct
          location, the character is greened out (encoded as 2 in the
          environment)

    The player continues to guess until they have either guessed the correct
    hidden word, or they have run out of guesses.

    The environment is structured in the following way:
        * Action Space: the action space is a length 5 MulitDiscrete where valid values
          are [0, 25], corresponding to characters [a, z].
        * Observation Space: the observation space is dict consisting of
          two objects:
          - board: The board is 6x5 Box corresponding to the history of
            guesses. At the start of the game, the board is filled entirely
            with -1 values, indicating no guess has been made. As the player
            guesses words, the rows will fill up with values in the range
            [0, 2] indicating whether the characters are missing in the
            hidden word, in the incorrect position, or in the correct position
			based on the most recent guess.
          - alphabet: the alphabet is a length 26 Box corresponding to the guess status
            for each letter in the alaphabet. As the start, all values are -1, as no letter
            has been used in a guess. As the player guesses words, the letters in the
            alphabet will change to values in the range [0, 2] indicating whether the
            characters are missing in the hidden word, in the incorrect position,
            or in the correct position.
    """

    def __init__(self):
        super(WordleEnv, self).__init__()
        self.words = WORDS
        self.action_space = spaces.Discrete(len(self.words))
        # self.action_space = spaces.MultiDiscrete([26] * WORD_LENGTH)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-1, high=2, shape=(GAME_LENGTH, WORD_LENGTH), dtype=int),
            'alphabet': spaces.Box(low=-1, high=2, shape=(26,), dtype=int)
        })
        self.guesses = []

    def step(self, action):
        assert self.action_space.contains(action)

        # Action must be a valid word
        if not action >= 0 and action < self.action_space.n :
            raise InvalidWordException(f'{action} does not correspond to a valid word')

        # update game board and alphabet tracking
        board_row_idx = GAME_LENGTH - self.guesses_left
        for idx, char in enumerate(WORDS[action]):

            if self.hidden_word[idx] == char:
                encoding = 2
            elif char in self.hidden_word:
                encoding = 1
            else:
                encoding = 0

            self.board[board_row_idx, idx] = encoding
            self.alphabet[char] = encoding

        # update guesses remaining tracker
        self.guesses_left -= 1

        # update previous guesses made
        self.guesses.append(WORDS[action])

        # check to see if game is over
        if all(self.board[board_row_idx, :] == 2):
            reward = 1.0
            done = True
        else:
            if self.guesses_left > 0:
                reward = 0.0
                done = False
            else:
                reward = -1.0
                done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return {'board': self.board, 'alphabet': self.alphabet}

    def reset(self, seed: Optional[int] = None):
        # super().reset(seed=seed)
        self.hidden_word = random.choice(WORDS)
        self.guesses_left = GAME_LENGTH
        self.board = np.negative(
            np.ones(shape=(GAME_LENGTH, WORD_LENGTH), dtype=int))
        self.alphabet = np.negative(np.ones(shape=(26,), dtype=int))
        self.guesses = []
        return self._get_obs()

    def render(self, mode="human"):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        print('###################################################')
        for i in range(len(self.guesses)):
            for j in range(WORD_LENGTH):
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
        print('###################################################')
        print()
