from enum import Enum


class ItemType(str, Enum):
    MODEL = "MODEL"
    AGENT = "AGENT"
    GENERAL = "GENERAL"

    def __str__(self) -> str:
        return self.value
