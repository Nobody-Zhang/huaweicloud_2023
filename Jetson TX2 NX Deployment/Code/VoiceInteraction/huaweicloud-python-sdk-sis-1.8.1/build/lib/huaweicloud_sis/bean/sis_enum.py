from enum import Enum


class WebsocketStatus(Enum):
    STATE_INIT = 0
    STATE_CONNECT_WAITING = 1
    STATE_CONNECTED = 2
    STATE_START_WAITING = 3
    STATE_START = 4
    STATE_END_WAITING = 5
    STATE_END = 6
    STATE_CLOSE = 7
    STATE_ERROR = 8
