"""Tests for data preprocessing."""

import pytest
from src.data.preprocessing import clean_text, format_instruction, create_chat_format


class TestCleanText:
    def test_clean_text_removes_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_clean_text_empty_string(self):
        assert clean_text("") == ""

    def test_clean_text_none(self):
        assert clean_text(None) == ""


class TestFormatInstruction:
    def test_format_with_input(self):
        result = format_instruction("Do this", "input data", "output data")
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result

    def test_format_without_input(self):
        result = format_instruction("Do this", "", "output data")
        assert "### Instruction:" in result
        assert "### Input:" not in result
        assert "### Response:" in result


class TestCreateChatFormat:
    def test_chat_format(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = create_chat_format(messages)
        assert "<|user|>" in result
        assert "<|assistant|>" in result
