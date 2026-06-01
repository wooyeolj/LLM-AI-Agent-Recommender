from enum import Enum
# 자주 사용하는 "MODEL", "AGENT", "GENERAL"를 타입으로 정의

class ItemType(str, Enum):
    MODEL = "MODEL"
    AGENT = "AGENT"
    GENERAL = "GENERAL"

    def __str__(self) -> str:
        return self.value
