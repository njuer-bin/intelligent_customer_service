import pytest
from datetime import datetime
from sme_guard.schemas import (
    Merchant,
    KnowledgeItem,
    Session,
    UnknownQuestion,
    Message,
)


class TestMerchant:
    def test_merchant_creation_with_required_fields(self):
        merchant = Merchant(name="测试店铺", industry="餐饮")
        assert merchant.name == "测试店铺"
        assert merchant.industry == "餐饮"
        assert merchant.id is not None
        assert isinstance(merchant.created_at, datetime)
        assert isinstance(merchant.updated_at, datetime)

    def test_merchant_missing_required_field_raises(self):
        with pytest.raises(Exception):
            Merchant(name="测试店铺")

    def test_merchant_update_timestamp(self):
        merchant = Merchant(name="测试店铺", industry="餐饮")
        old_updated = merchant.updated_at
        merchant.update_timestamp()
        assert merchant.updated_at > old_updated


class TestKnowledgeItem:
    def test_knowledge_item_creation(self):
        item = KnowledgeItem(
            merchant_id="m1",
            title="营业时间",
            content="周一至周日 9:00-22:00",
            source="manual",
        )
        assert item.merchant_id == "m1"
        assert item.title == "营业时间"
        assert item.content == "周一至周日 9:00-22:00"
        assert item.source == "manual"
        assert item.id is not None
        assert isinstance(item.created_at, datetime)

    def test_knowledge_item_missing_merchant_id_raises(self):
        with pytest.raises(Exception):
            KnowledgeItem(title="标题", content="内容", source="manual")


class TestSession:
    def test_session_creation(self):
        session = Session(merchant_id="m1")
        assert session.merchant_id == "m1"
        assert session.messages == []
        assert session.id is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_add_message(self):
        session = Session(merchant_id="m1")
        session.add_message("user", "你好")
        session.add_message("assistant", "您好！")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "你好"
        assert session.messages[1].role == "assistant"
        assert session.updated_at > session.created_at

    def test_get_recent_messages(self):
        session = Session(merchant_id="m1")
        for i in range(15):
            session.add_message("user", f"消息{i}")
        recent = session.get_recent_messages(10)
        assert len(recent) == 10
        assert recent[0].content == "消息5"

    def test_clear_session(self):
        session = Session(merchant_id="m1")
        session.add_message("user", "测试")
        session.clear()
        assert session.messages == []
        assert session.updated_at > session.created_at


class TestUnknownQuestion:
    def test_unknown_question_creation(self):
        uq = UnknownQuestion(
            merchant_id="m1",
            question="你们有外卖吗？",
            context="用户咨询外卖业务",
        )
        assert uq.merchant_id == "m1"
        assert uq.question == "你们有外卖吗？"
        assert uq.context == "用户咨询外卖业务"
        assert uq.id is not None
        assert isinstance(uq.created_at, datetime)

    def test_unknown_question_missing_fields_raises(self):
        with pytest.raises(Exception):
            UnknownQuestion(merchant_id="m1", question="测试")


class TestMessage:
    def test_message_creation(self):
        msg = Message(role="user", content="测试消息")
        assert msg.role == "user"
        assert msg.content == "测试消息"
        assert isinstance(msg.timestamp, datetime)