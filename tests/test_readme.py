import pathlib

def test_readme_contains_architecture_diagram():
    readme_path = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    assert readme_path.is_file(), "README.md does not exist"
    content = readme_path.read_text()
    assert "Architecture diagram" in content, "README.md does not contain 'Architecture diagram'"
