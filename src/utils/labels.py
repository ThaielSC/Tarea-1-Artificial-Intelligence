from enum import Enum
from typing import Dict, List, Optional

class Category(Enum):
    NUMBERS = "numbers"
    ANIMALS = "animals"
    DIRECTIONS = "directions"
    COMMANDS = "commands"
    OBJECTS = "objects"
    NAMES = "names"
    EMOTIONS = "emotions"

class LabelManager:
    _MAPPING: Dict[Category, List[str]] = {
        Category.NUMBERS: ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
        Category.ANIMALS: ["bird", "cat", "dog"],
        Category.DIRECTIONS: ["up", "down", "left", "right"],
        Category.COMMANDS: ["go", "stop", "on", "off", "yes", "no"],
        Category.OBJECTS: ["bed", "house", "tree"],
        Category.NAMES: ["marvin", "sheila"],
        Category.EMOTIONS: ["happy", "wow"]
    }

    _WORD_TO_CAT: Dict[str, Category] = {
        word: cat for cat, words in _MAPPING.items() for word in words
    }

    _CAT_TO_IDX: Dict[Category, int] = {
        cat: i for i, cat in enumerate(Category)
    }

    _IDX_TO_CAT: Dict[int, Category] = {
        i: cat for cat, i in _CAT_TO_IDX.items()
    }

    @classmethod
    def get_category_of_word(cls, word: str) -> Optional[Category]:
        return cls._WORD_TO_CAT.get(word)

    @classmethod
    def get_category_idx(cls, category: Category) -> int:
        return cls._CAT_TO_IDX[category]

    @classmethod
    def get_category_from_idx(cls, idx: int) -> Category:
        return cls._IDX_TO_CAT[idx]

    @classmethod
    def get_num_categories(cls) -> int:
        return len(Category)

    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [cat.value for cat in Category]
