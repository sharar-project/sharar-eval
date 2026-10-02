import pytest


@pytest.fixture(scope="session")
def snapshot():
    """The pinned Qwen3.8-27B snapshot (tokenizer and chat template); tests that need it skip without it."""
    from evals.speed import model

    try:
        return model.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
