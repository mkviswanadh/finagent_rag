"""Tests for archive.py — timestamped snapshotting of result files before they're overwritten."""

from __future__ import annotations

from finagent.results.archive import archive_file


def test_returns_none_when_source_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    result = archive_file(missing, category="pilot", archive_dir=tmp_path / "archive")
    assert result is None


def test_copies_file_into_category_subdirectory(tmp_path):
    source = tmp_path / "pilot_run_report.json"
    source.write_text('{"a": 1}', encoding="utf-8")
    archive_dir = tmp_path / "archive"

    result = archive_file(source, category="pilot", archive_dir=archive_dir)

    assert result is not None
    assert result.exists()
    assert result.parent == archive_dir / "pilot"
    assert result.read_text(encoding="utf-8") == '{"a": 1}'


def test_does_not_modify_or_remove_the_source(tmp_path):
    source = tmp_path / "report.json"
    source.write_text("original", encoding="utf-8")

    archive_file(source, category="pilot", archive_dir=tmp_path / "archive")

    assert source.exists()
    assert source.read_text(encoding="utf-8") == "original"


def test_archived_filename_includes_timestamp_and_original_stem(tmp_path):
    source = tmp_path / "pilot_run_report.json"
    source.write_text("x", encoding="utf-8")

    result = archive_file(source, category="pilot", archive_dir=tmp_path / "archive")

    assert result.name.startswith("pilot_run_report_")
    assert result.suffix == ".json"
    assert result.name != source.name


def test_two_archives_of_different_categories_do_not_collide(tmp_path):
    source = tmp_path / "report.json"
    source.write_text("x", encoding="utf-8")
    archive_dir = tmp_path / "archive"

    pilot_copy = archive_file(source, category="pilot", archive_dir=archive_dir)
    full_run_copy = archive_file(source, category="full_run", archive_dir=archive_dir)

    assert pilot_copy.parent != full_run_copy.parent
    assert pilot_copy.exists() and full_run_copy.exists()
