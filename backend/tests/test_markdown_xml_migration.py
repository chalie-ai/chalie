import sqlite3
import pytest
from services.markdown_xml_migration import markdown_to_xml, run_if_needed


@pytest.mark.unit
class TestMarkdownToXml:
    def test_plain_paragraph(self):
        assert markdown_to_xml("hello world") == "<p>hello world</p>"

    def test_two_paragraphs(self):
        result = markdown_to_xml("first\n\nsecond")
        assert result == "<p>first</p><p>second</p>"

    def test_bold_double_asterisk(self):
        assert markdown_to_xml("a **bold** b") == "<p>a <b>bold</b> b</p>"

    def test_bold_double_underscore(self):
        assert markdown_to_xml("a __bold__ b") == "<p>a <b>bold</b> b</p>"

    def test_italic_single_asterisk(self):
        assert markdown_to_xml("a *italic* b") == "<p>a <i>italic</i> b</p>"

    def test_italic_single_underscore(self):
        assert markdown_to_xml("a _italic_ b") == "<p>a <i>italic</i> b</p>"

    def test_inline_code(self):
        assert markdown_to_xml("use `foo()` here") == "<p>use <code>foo()</code> here</p>"

    def test_fenced_code(self):
        result = markdown_to_xml("```\nx = 1\n```")
        assert result == "<code>x = 1</code>"

    def test_fenced_code_with_lang(self):
        result = markdown_to_xml("```python\nx = 1\n```")
        assert result == "<code>x = 1</code>"

    def test_h1(self):
        assert markdown_to_xml("# Title") == "<h1>Title</h1>"

    def test_h2_collapses_to_h1(self):
        assert markdown_to_xml("## Title") == "<h1>Title</h1>"

    def test_h3_collapses_to_h1(self):
        assert markdown_to_xml("### Title") == "<h1>Title</h1>"

    def test_unordered_list(self):
        result = markdown_to_xml("- one\n- two\n- three")
        assert result == "<ul><li>one</li><li>two</li><li>three</li></ul>"

    def test_ordered_list_collapses_to_unordered(self):
        result = markdown_to_xml("1. one\n2. two")
        assert result == "<ul><li>one</li><li>two</li></ul>"

    def test_link(self):
        result = markdown_to_xml("see [docs](https://x.com)")
        assert result == '<p>see <a href="https://x.com">docs</a></p>'

    def test_image(self):
        result = markdown_to_xml("![cat](https://x.com/c.png)")
        assert result == '<img src="https://x.com/c.png" alt="cat"/>'

    def test_blockquote_flattens(self):
        # Best-effort: blockquote becomes italic paragraph
        result = markdown_to_xml("> quoted text")
        assert result == "<p><i>quoted text</i></p>"

    def test_strikethrough_drops(self):
        # No ~~ tag in our set — strip the tildes, keep text
        assert markdown_to_xml("a ~~strike~~ b") == "<p>a strike b</p>"

    def test_horizontal_rule_drops(self):
        result = markdown_to_xml("before\n\n---\n\nafter")
        assert result == "<p>before</p><p>after</p>"

    def test_table_flattens_to_paragraphs(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |"
        result = markdown_to_xml(md)
        # Header row + data row, each as a paragraph with cells space-separated
        assert result == "<p>a b</p><p>1 2</p>"

    def test_escapes_xml_special_chars(self):
        assert markdown_to_xml("a < b > c & d") == "<p>a &lt; b &gt; c &amp; d</p>"

    def test_already_xml_returns_unchanged(self):
        # Sentinel: if input already starts with a tag, treat as XML and pass through
        assert markdown_to_xml("<p>already</p>") == "<p>already</p>"

    def test_empty(self):
        assert markdown_to_xml("") == ""

    def test_combined_features(self):
        md = "# Title\n\nThis is **bold** and *italic*.\n\n- one\n- two\n\n```\ncode\n```"
        result = markdown_to_xml(md)
        assert result == (
            "<h1>Title</h1>"
            "<p>This is <b>bold</b> and <i>italic</i>.</p>"
            "<ul><li>one</li><li>two</li></ul>"
            "<code>code</code>"
        )


@pytest.mark.unit
class TestRunIfNeeded:
    def _build_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE transcript (
                id INTEGER PRIMARY KEY,
                role TEXT,
                content TEXT,
                xml_migrated INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        return conn

    def test_converts_unmigrated_rows(self):
        conn = self._build_db()
        conn.execute(
            "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, ?, 0)",
            ("assistant", "**hello** world"),
        )
        conn.commit()

        run_if_needed(conn)

        row = conn.execute("SELECT content, xml_migrated FROM transcript").fetchone()
        assert row[0] == "<p><b>hello</b> world</p>"
        assert row[1] == 1

    def test_skips_already_migrated(self):
        conn = self._build_db()
        conn.execute(
            "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, ?, 1)",
            ("assistant", "**unchanged**"),
        )
        conn.commit()

        run_if_needed(conn)

        row = conn.execute("SELECT content, xml_migrated FROM transcript").fetchone()
        assert row[0] == "**unchanged**"  # not touched
        assert row[1] == 1

    def test_idempotent_second_run(self):
        conn = self._build_db()
        conn.execute(
            "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, ?, 0)",
            ("assistant", "hi"),
        )
        conn.commit()

        run_if_needed(conn)
        first = conn.execute("SELECT content FROM transcript").fetchone()[0]

        run_if_needed(conn)
        second = conn.execute("SELECT content FROM transcript").fetchone()[0]

        assert first == second == "<p>hi</p>"

    def test_non_assistant_rows_skip_content_conversion(self):
        # Only assistant rows render through the XML markup pipeline. Other
        # roles (user, tool, proactive_thought, scheduled, …) are rendered as
        # plaintext or fed back to the model verbatim — wrapping them in <p>
        # would leak literal tags into the UI or mangle structured tool output.
        conn = self._build_db()
        non_assistant_roles = (
            ("user", "hello **world**"),
            ("tool", '{"result": "ok"}'),
            ("proactive_thought", "should I check this?"),
            ("scheduled", "reminder: drink water"),
            ("tool_synthesis", "summary line"),
            ("goal_pursuit", "investigate X"),
        )
        for role, content in non_assistant_roles:
            conn.execute(
                "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, ?, 0)",
                (role, content),
            )
        conn.commit()

        run_if_needed(conn)

        rows = conn.execute(
            "SELECT role, content, xml_migrated FROM transcript ORDER BY id"
        ).fetchall()
        for (role, original), (got_role, got_content, got_flag) in zip(
            non_assistant_roles, rows, strict=True
        ):
            assert got_role == role
            assert got_content == original  # untouched
            assert got_flag == 1  # sentinel still flipped so we don't retry

    def test_handles_null_content(self):
        conn = self._build_db()
        conn.execute(
            "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, NULL, 0)",
            ("system",),
        )
        conn.commit()

        run_if_needed(conn)
        row = conn.execute("SELECT content, xml_migrated FROM transcript").fetchone()
        assert row[0] is None
        assert row[1] == 1  # still marked migrated to avoid retry storms

    def test_returns_count(self):
        conn = self._build_db()
        for content in ("**a**", "_b_", "c"):
            conn.execute(
                "INSERT INTO transcript (role, content, xml_migrated) VALUES (?, ?, 0)",
                ("assistant", content),
            )
        conn.commit()

        count = run_if_needed(conn)
        assert count == 3
