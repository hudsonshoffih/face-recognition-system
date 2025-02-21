from dataclasses import dataclass
from datetime import datetime

@dataclass
class Face:
    id: int
    name: str
    face_encoding: bytes
    created_at: datetime

@dataclass
class Entry:
    id: int
    face_id: int
    entry_time: datetime