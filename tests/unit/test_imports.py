import pytest


def test_import_sme_guard():
    import sme_guard
    assert sme_guard is not None


def test_import_all_submodules():
    modules = [
        "sme_guard.llm_client",
        "sme_guard.agents",
        "sme_guard.agents.router",
        "sme_guard.tools",
        "sme_guard.tools.price_lookup",
        "sme_guard.tools.avail_lookup",
        "sme_guard.rag",
        "sme_guard.rag.ingest",
        "sme_guard.rag.retrieve",
        "sme_guard.rag.vector_db",
        "sme_guard.memory",
        "sme_guard.memory.session",
        "sme_guard.store",
        "sme_guard.store.unknowns",
        "sme_guard.ui",
        "sme_guard.ui.gradio_app",
        "sme_guard.schemas",
    ]
    for mod in modules:
        __import__(mod)


def test_import_schemas_models():
    from sme_guard.schemas import (
        Merchant,
        KnowledgeItem,
        Session,
        UnknownQuestion,
        Message,
    )
    assert all([
        Merchant,
        KnowledgeItem,
        Session,
        UnknownQuestion,
        Message,
    ])