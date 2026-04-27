"""
Unit tests for BlocksRenderService.

Coverage:
  from_markdown:
    - empty / whitespace-only input
    - plain paragraphs (single and multi)
    - inline markdown preserved in text blocks
    - ATX headings (H1–H6), with inline markdown
    - fenced code blocks (with/without language, whitespace, blank lines inside,
      surrounding text, multiple in sequence)
    - unordered lists (- and * syntax), inline markdown in items
    - ordered lists
    - paragraph/list/paragraph interleaving
    - two separate lists with paragraph between them
    - GFM tables (simple and varying column counts), table + text
    - horizontal rules (---, ***, ___)
    - mixed real-world LLM response
    - leading/trailing whitespace stripping
    - only a code block
  from_tool_result:
    - delegates 'text' to from_markdown
    - 'error' field → system message block at level "error"
    - empty dict → empty list
  from_system_message:
    - basic info message
    - error level
  from_actions:
    - list of buttons → actions block
    - empty list → empty list
"""

import pytest
from services.blocks_render_service import BlocksRenderService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type(block: dict) -> str:
    return block["type"]


# ---------------------------------------------------------------------------
# from_markdown — empty / whitespace
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownEmpty:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_empty_string(self):
        assert self.svc.from_markdown("") == []

    def test_whitespace_only_spaces(self):
        assert self.svc.from_markdown("   ") == []

    def test_whitespace_only_newlines(self):
        assert self.svc.from_markdown("\n\n\n") == []

    def test_whitespace_mixed(self):
        assert self.svc.from_markdown("  \n  \n  ") == []


# ---------------------------------------------------------------------------
# from_markdown — plain paragraphs
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownParagraphs:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_single_paragraph(self):
        blocks = self.svc.from_markdown("Hello world")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Hello world" in blocks[0]["content"]
        assert blocks[0]["format"] == "markdown"

    def test_multiple_paragraphs_separated_by_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 3
        assert all(_type(b) == "text" for b in blocks)
        assert "First paragraph." in blocks[0]["content"]
        assert "Second paragraph." in blocks[1]["content"]
        assert "Third paragraph." in blocks[2]["content"]

    def test_inline_markdown_preserved_in_text_block_bold(self):
        blocks = self.svc.from_markdown("This is **bold** text")
        assert len(blocks) == 1
        assert "**bold**" in blocks[0]["content"]

    def test_inline_markdown_preserved_italic(self):
        blocks = self.svc.from_markdown("This is *italic* text")
        assert "italic" in blocks[0]["content"]
        assert blocks[0]["format"] == "markdown"

    def test_inline_markdown_preserved_code(self):
        blocks = self.svc.from_markdown("Use `print()` to display output")
        assert "`print()`" in blocks[0]["content"]

    def test_inline_markdown_preserved_link(self):
        blocks = self.svc.from_markdown("See [the docs](https://example.com) for details")
        assert "[the docs]" in blocks[0]["content"]
        assert "https://example.com" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# from_markdown — headers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownHeaders:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_h1(self):
        blocks = self.svc.from_markdown("# H1 Title")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "header"
        assert blocks[0]["level"] == 1
        assert blocks[0]["content"] == "H1 Title"

    def test_h2(self):
        blocks = self.svc.from_markdown("## H2 Title")
        assert blocks[0]["level"] == 2

    def test_h3(self):
        blocks = self.svc.from_markdown("### H3 Title")
        assert blocks[0]["level"] == 3

    def test_h6(self):
        blocks = self.svc.from_markdown("###### H6 Title")
        assert blocks[0]["level"] == 6

    def test_header_followed_by_paragraph(self):
        text = "# Section\n\nSome body text."
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert "header" in types
        assert "text" in types
        header = next(b for b in blocks if _type(b) == "header")
        assert header["content"] == "Section"

    def test_header_with_inline_markdown(self):
        blocks = self.svc.from_markdown("## Hello **world**")
        assert blocks[0]["type"] == "header"
        assert "Hello" in blocks[0]["content"]
        assert "**world**" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# from_markdown — fenced code blocks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownCodeBlocks:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_code_block_with_language(self):
        text = "```python\nprint('hello')\n```"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["language"] == "python"
        assert "print('hello')" in blocks[0]["content"]

    def test_code_block_without_language(self):
        text = "```\nsome code here\n```"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["language"] == ""

    def test_code_block_preserves_internal_whitespace(self):
        text = "```python\ndef foo():\n    return 42\n```"
        blocks = self.svc.from_markdown(text)
        assert "    return 42" in blocks[0]["content"]

    def test_code_block_preserves_indentation(self):
        text = "```\n    indented\n        more indented\n```"
        blocks = self.svc.from_markdown(text)
        content = blocks[0]["content"]
        assert "    indented" in content
        assert "        more indented" in content

    def test_code_block_with_blank_lines_inside(self):
        text = "```javascript\nfunction foo() {\n\n    return 1;\n}\n```"
        blocks = self.svc.from_markdown(text)
        assert blocks[0]["type"] == "code"
        content = blocks[0]["content"]
        assert "function foo()" in content
        assert "return 1;" in content

    def test_text_before_and_after_code_block(self):
        text = "Before the code.\n\n```python\nx = 1\n```\n\nAfter the code."
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert types.count("code") == 1
        assert types.count("text") == 2
        code_idx = types.index("code")
        assert types[0] == "text"
        assert types[code_idx] == "code"
        assert types[-1] == "text"

    def test_multiple_code_blocks_in_sequence(self):
        text = "```python\na = 1\n```\n\n```bash\necho hi\n```"
        blocks = self.svc.from_markdown(text)
        code_blocks = [b for b in blocks if _type(b) == "code"]
        assert len(code_blocks) == 2
        assert code_blocks[0]["language"] == "python"
        assert code_blocks[1]["language"] == "bash"


# ---------------------------------------------------------------------------
# from_markdown — lists
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownLists:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_unordered_list_dash_syntax(self):
        text = "- First\n- Second\n- Third"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "list"
        assert blocks[0]["style"] == "unordered"
        assert blocks[0]["items"] == ["First", "Second", "Third"]

    def test_unordered_list_asterisk_syntax(self):
        text = "* Apple\n* Banana\n* Cherry"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "list"
        assert blocks[0]["style"] == "unordered"
        assert "Apple" in blocks[0]["items"]

    def test_ordered_list(self):
        text = "1. First\n2. Second\n3. Third"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "list"
        assert blocks[0]["style"] == "ordered"
        assert blocks[0]["items"] == ["First", "Second", "Third"]

    def test_list_items_preserve_inline_markdown(self):
        text = "- Item with **bold**\n- Item with `code`"
        blocks = self.svc.from_markdown(text)
        assert "**bold**" in blocks[0]["items"][0]
        assert "`code`" in blocks[0]["items"][1]

    def test_paragraph_followed_by_list(self):
        text = "Intro text.\n\n- Item one\n- Item two"
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert "text" in types
        assert "list" in types
        assert types.index("text") < types.index("list")

    def test_two_separate_lists_with_paragraph_between(self):
        text = "- List A item\n\nMiddle paragraph.\n\n1. List B item"
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert types.count("list") == 2
        assert types.count("text") == 1
        # Order: list, text, list
        assert types[0] == "list"
        assert types[1] == "text"
        assert types[2] == "list"
        # Styles
        list_blocks = [b for b in blocks if _type(b) == "list"]
        assert list_blocks[0]["style"] == "unordered"
        assert list_blocks[1]["style"] == "ordered"


# ---------------------------------------------------------------------------
# from_markdown — tables
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownTables:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_simple_two_column_table(self):
        text = "| Name | Value |\n|------|-------|\n| foo  | bar   |\n| baz  | qux   |"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"
        assert blocks[0]["headers"] == ["Name", "Value"]
        assert len(blocks[0]["rows"]) == 2
        assert "foo" in blocks[0]["rows"][0]
        assert "bar" in blocks[0]["rows"][0]

    def test_table_with_three_columns(self):
        text = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
        blocks = self.svc.from_markdown(text)
        assert blocks[0]["type"] == "table"
        assert len(blocks[0]["headers"]) == 3
        assert blocks[0]["rows"][0] == ["1", "2", "3"]

    def test_table_cells_are_trimmed(self):
        text = "|  Name  |  Value  |\n|--------|----------|\n|  Alice |  30      |"
        blocks = self.svc.from_markdown(text)
        assert blocks[0]["headers"] == ["Name", "Value"]
        assert blocks[0]["rows"][0] == ["Alice", "30"]

    def test_table_followed_by_text(self):
        text = "| X | Y |\n|---|---|\n| a | b |\n\nFollowing paragraph."
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert types[0] == "table"
        assert types[-1] == "text"


# ---------------------------------------------------------------------------
# from_markdown — dividers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownDividers:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_divider_dashes(self):
        blocks = self.svc.from_markdown("---")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "divider"

    def test_divider_asterisks(self):
        blocks = self.svc.from_markdown("***")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "divider"

    def test_divider_underscores(self):
        blocks = self.svc.from_markdown("___")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "divider"

    def test_divider_longer_dashes(self):
        blocks = self.svc.from_markdown("-------")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "divider"

    def test_divider_between_paragraphs(self):
        text = "Above.\n\n---\n\nBelow."
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert "divider" in types
        assert types.count("text") == 2


# ---------------------------------------------------------------------------
# from_markdown — mixed / real-world
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromMarkdownMixed:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_real_world_llm_response(self):
        text = (
            "# Summary\n\n"
            "Here is a brief overview of the findings.\n\n"
            "## Details\n\n"
            "The analysis revealed three key points:\n\n"
            "- Point one with **emphasis**\n"
            "- Point two mentioning `config.json`\n"
            "- Point three\n\n"
            "```python\ndef example():\n    return True\n```\n\n"
            "See the table below:\n\n"
            "| Key | Value |\n|-----|-------|\n| a   | 1     |\n"
        )
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert "header" in types
        assert "text" in types
        assert "list" in types
        assert "code" in types
        assert "table" in types

    def test_leading_trailing_whitespace_stripped(self):
        blocks = self.svc.from_markdown("\n\n  Hello world  \n\n")
        assert len(blocks) >= 1
        text_blocks = [b for b in blocks if _type(b) == "text"]
        assert len(text_blocks) == 1
        # Content should not have leading/trailing paragraph whitespace
        assert blocks[0]["content"].strip() == blocks[0]["content"]

    def test_only_code_block(self):
        text = "```bash\necho hello\n```"
        blocks = self.svc.from_markdown(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["language"] == "bash"

    def test_header_immediately_followed_by_code_block(self):
        text = "# Usage\n```python\nprint('hi')\n```"
        blocks = self.svc.from_markdown(text)
        types = [_type(b) for b in blocks]
        assert types[0] == "header"
        assert "code" in types


# ---------------------------------------------------------------------------
# from_tool_result
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromToolResult:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_text_field_delegates_to_from_markdown(self):
        result = {"text": "# Title\n\nSome body text."}
        blocks = self.svc.from_tool_result(result)
        types = [_type(b) for b in blocks]
        assert "header" in types
        assert "text" in types

    def test_text_field_plain_paragraph(self):
        result = {"text": "Simple tool output."}
        blocks = self.svc.from_tool_result(result)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Simple tool output." in blocks[0]["content"]

    def test_error_field_yields_system_message_block(self):
        result = {"error": "Something went wrong."}
        blocks = self.svc.from_tool_result(result)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["format"] == "plain"
        assert blocks[0]["level"] == "error"
        assert "Something went wrong." in blocks[0]["content"]

    def test_error_takes_priority_over_text(self):
        result = {"error": "Failure", "text": "This should be ignored"}
        blocks = self.svc.from_tool_result(result)
        assert len(blocks) == 1
        assert blocks[0]["level"] == "error"

    def test_empty_dict_returns_empty_list(self):
        assert self.svc.from_tool_result({}) == []

    def test_empty_text_field_returns_empty_list(self):
        assert self.svc.from_tool_result({"text": ""}) == []

    def test_empty_text_whitespace_returns_empty_list(self):
        assert self.svc.from_tool_result({"text": "   "}) == []


# ---------------------------------------------------------------------------
# from_system_message
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromSystemMessage:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_basic_info_message(self):
        blocks = self.svc.from_system_message("Operation completed.")
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "text"
        assert block["content"] == "Operation completed."
        assert block["format"] == "plain"
        assert block["level"] == "info"

    def test_error_level(self):
        blocks = self.svc.from_system_message("Something failed.", level="error")
        assert blocks[0]["level"] == "error"

    def test_warning_level(self):
        blocks = self.svc.from_system_message("Watch out.", level="warning")
        assert blocks[0]["level"] == "warning"

    def test_default_level_is_info(self):
        blocks = self.svc.from_system_message("Hello")
        assert blocks[0]["level"] == "info"

    def test_message_content_preserved(self):
        msg = "User@host: permission denied (publickey)."
        blocks = self.svc.from_system_message(msg)
        assert blocks[0]["content"] == msg


# ---------------------------------------------------------------------------
# from_actions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFromActions:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_action_buttons_returned_as_actions_block(self):
        actions = [
            {"label": "Confirm", "skill": "confirm", "payload": {"id": 1}},
            {"label": "Cancel",  "skill": "cancel",  "payload": {}},
        ]
        blocks = self.svc.from_actions(actions)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        assert blocks[0]["buttons"] == actions

    def test_single_button(self):
        actions = [{"label": "OK"}]
        blocks = self.svc.from_actions(actions)
        assert blocks[0]["type"] == "actions"
        assert len(blocks[0]["buttons"]) == 1
        assert blocks[0]["buttons"][0]["label"] == "OK"

    def test_empty_actions_list_returns_empty_list(self):
        assert self.svc.from_actions([]) == []

    def test_buttons_list_is_the_same_objects(self):
        btn = {"label": "Go", "skill": "navigate"}
        blocks = self.svc.from_actions([btn])
        assert blocks[0]["buttons"][0] is btn


# ---------------------------------------------------------------------------
# Block directives — fenced code blocks parsed as typed blocks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBlockDirectives:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_metric_directive(self):
        md = '```metric\n{"label":"CPU","value":"73%","trend":"up"}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "metric"
        assert blocks[0]["label"] == "CPU"
        assert blocks[0]["value"] == "73%"
        assert blocks[0]["trend"] == "up"

    def test_card_directive(self):
        md = '```card\n{"title":"Summary","body":"All good","accent":"cyan"}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "card"
        assert blocks[0]["title"] == "Summary"
        assert blocks[0]["accent"] == "cyan"

    def test_chart_directive(self):
        md = '```chart\n{"series":[{"label":"S1","values":[1,2,3]}],"chartType":"bar"}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "chart"
        assert blocks[0]["series"][0]["values"] == [1, 2, 3]

    def test_progress_directive(self):
        md = '```progress\n{"label":"Upload","value":42}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "progress"
        assert blocks[0]["value"] == 42

    def test_timeline_directive(self):
        md = '```timeline\n{"events":[{"label":"Step 1","status":"done"},{"label":"Step 2","status":"active"}]}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "timeline"
        assert len(blocks[0]["events"]) == 2

    def test_alert_directive(self):
        md = '```alert\n{"message":"Watch out","variant":"warning"}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "alert"
        assert blocks[0]["variant"] == "warning"

    def test_invalid_json_falls_back_to_code_block(self):
        md = '```metric\nnot valid json\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["language"] == "metric"

    def test_json_array_falls_back_to_code_block(self):
        md = '```metric\n[1, 2, 3]\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"

    def test_non_directive_language_stays_as_code(self):
        md = '```python\nprint("hello")\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["language"] == "python"

    def test_directive_mixed_with_text(self):
        md = 'Here are the results:\n\n```metric\n{"label":"Score","value":"95"}\n```\n\nLooks great!'
        blocks = self.svc.from_markdown(md)
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "metric"
        assert blocks[2]["type"] == "text"

    def test_multiple_directives(self):
        md = '```metric\n{"label":"A","value":"1"}\n```\n```metric\n{"label":"B","value":"2"}\n```'
        blocks = self.svc.from_markdown(md)
        assert len(blocks) == 2
        assert blocks[0]["label"] == "A"
        assert blocks[1]["label"] == "B"


# ---------------------------------------------------------------------------
# Validation — new block types
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateNewTypes:
    def setup_method(self):
        self.svc = BlocksRenderService()

    def test_metric_valid(self):
        errors = self.svc.validate([{"type": "metric", "label": "CPU", "value": "73%"}])
        assert errors == []

    def test_metric_missing_label(self):
        errors = self.svc.validate([{"type": "metric", "value": "73%"}])
        assert any("label" in e for e in errors)

    def test_metric_missing_value(self):
        errors = self.svc.validate([{"type": "metric", "label": "CPU"}])
        assert any("value" in e for e in errors)

    def test_card_valid(self):
        errors = self.svc.validate([{"type": "card", "title": "Test"}])
        assert errors == []

    def test_card_missing_title(self):
        errors = self.svc.validate([{"type": "card", "body": "text"}])
        assert any("title" in e for e in errors)

    def test_chart_valid(self):
        errors = self.svc.validate([{"type": "chart", "series": [{"label": "S", "values": [1]}]}])
        assert errors == []

    def test_chart_missing_series(self):
        errors = self.svc.validate([{"type": "chart"}])
        assert any("series" in e for e in errors)

    def test_progress_valid(self):
        errors = self.svc.validate([{"type": "progress", "label": "X", "value": 50}])
        assert errors == []

    def test_progress_value_not_number(self):
        errors = self.svc.validate([{"type": "progress", "label": "X", "value": "50"}])
        assert any("number" in e for e in errors)

    def test_timeline_valid(self):
        errors = self.svc.validate([{"type": "timeline", "events": [{"label": "Step"}]}])
        assert errors == []

    def test_timeline_missing_events(self):
        errors = self.svc.validate([{"type": "timeline"}])
        assert any("events" in e for e in errors)

    def test_previously_unknown_types_now_known(self):
        """badge, alert, columns, section, etc. should no longer trigger 'unknown type'."""
        for btype in ("badge", "alert", "columns", "section", "tabs", "loading", "thought"):
            errors = self.svc.validate([{"type": btype}])
            assert not any("unknown type" in e for e in errors), f"{btype} flagged as unknown"


# ---------------------------------------------------------------------------
# Rich render skill
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRichRenderSkill:
    def test_returns_reference_string(self):
        from abilities.rich_render import RichRenderAbility
        ability = RichRenderAbility()
        result = ability.execute("test", {}, None)
        text = result["text"]
        assert isinstance(text, str)
        assert "metric" in text
        assert "card" in text
        assert "chart" in text
        assert "timeline" in text
        assert "progress" in text

    def test_tool_schema_present(self):
        from abilities.rich_render import RichRenderAbility
        assert RichRenderAbility.NAME == "rich_render"
        assert isinstance(RichRenderAbility.INPUT_SCHEMA, dict)
