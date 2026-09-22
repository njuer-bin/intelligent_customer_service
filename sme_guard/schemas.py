from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=now_utc)


class Merchant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    industry: str
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    def update_timestamp(self):
        self.updated_at = now_utc()


class KnowledgeItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    merchant_id: str
    title: str
    content: str
    source: str
    created_at: datetime = Field(default_factory=now_utc)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    merchant_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        self.updated_at = now_utc()

    def get_recent_messages(self, k: int = 10) -> List[Message]:
        return self.messages[-k:]

    def clear(self):
        self.messages.clear()
        self.updated_at = now_utc()


class UnknownQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    merchant_id: str
    question: str
    context: str
    created_at: datetime = Field(default_factory=now_utc)