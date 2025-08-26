from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
        # 新增：可用字符串 "random_with_state" 或整数 4
    RANDOM_WITH_STATE = 4
    # 新增：可用字符串 "globalscheduleonline" 或整数 5
    GLOBALSCHEDULEONLINE = 5
    PPOONLINE = 6
