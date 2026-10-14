"""The sharar-eval command: every command names a runnable module, arguments reach it, exits pass through."""
import importlib.util

from evals import cli


def test_every_command_names_a_runnable_module():
    for name, (module, _, _) in cli.COMMANDS.items():
        spec = importlib.util.find_spec(module)
        assert spec is not None, name
        if spec.submodule_search_locations:  # a package runs its __main__
            assert importlib.util.find_spec(module + ".__main__") is not None, name


def test_help_lists_every_command(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert all(f"  {name} " in out for name in cli.COMMANDS)


def test_arguments_reach_the_module_and_exits_pass_through(monkeypatch, capsys):
    seen = {}

    def run_module(module, run_name):
        import sys
        seen["module"], seen["argv"] = module, list(sys.argv)
        raise SystemExit("run.json differs")

    monkeypatch.setattr(cli.runpy, "run_module", run_module)
    assert cli.main(["compare", "a", "b", "--detail"]) == 1
    assert seen == {"module": "evals.suite", "argv": ["sharar-eval compare", "compare", "a", "b", "--detail"]}
    assert "run.json differs" in capsys.readouterr().err
    monkeypatch.setattr(cli.runpy, "run_module", lambda module, run_name: (_ for _ in ()).throw(SystemExit(3)))
    assert cli.main(["speed"]) == 3


def test_an_unknown_command_is_an_error(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err
