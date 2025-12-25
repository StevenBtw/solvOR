"""Tests for shared types."""

from solvor.types import Progress, Result, Status


class TestStatus:
    def test_status_values(self):
        assert Status.OPTIMAL == 1
        assert Status.FEASIBLE == 2
        assert Status.INFEASIBLE == 3
        assert Status.UNBOUNDED == 4
        assert Status.MAX_ITER == 5


class TestResult:
    def test_result_ok_optimal(self):
        result = Result(solution=[1, 2], objective=10.0, status=Status.OPTIMAL)
        assert result.ok is True

    def test_result_ok_feasible(self):
        result = Result(solution=[1, 2], objective=10.0, status=Status.FEASIBLE)
        assert result.ok is True

    def test_result_not_ok_infeasible(self):
        result = Result(solution=None, objective=0, status=Status.INFEASIBLE)
        assert result.ok is False

    def test_result_not_ok_unbounded(self):
        result = Result(solution=None, objective=0, status=Status.UNBOUNDED)
        assert result.ok is False

    def test_result_not_ok_max_iter(self):
        result = Result(solution=[1], objective=5.0, status=Status.MAX_ITER)
        assert result.ok is False

    def test_result_log_returns_self(self):
        result = Result(solution=[1, 2], objective=10.0, status=Status.OPTIMAL)
        returned = result.log("prefix: ")
        assert returned is result

    def test_result_log_with_debug(self, capsys, monkeypatch):
        monkeypatch.setenv("DEBUG", "1")
        # Need to reimport to pick up the env var
        import importlib

        import solvor.types

        importlib.reload(solvor.types)

        result = solvor.types.Result(solution=[1, 2], objective=10.0, iterations=5, status=solvor.types.Status.OPTIMAL)
        result.log("test: ")
        captured = capsys.readouterr()
        assert "OPTIMAL" in captured.out
        assert "obj=10.0" in captured.out
        assert "iter=5" in captured.out

        # Cleanup: reload without DEBUG
        monkeypatch.delenv("DEBUG", raising=False)
        importlib.reload(solvor.types)

    def test_result_log_with_error(self, capsys, monkeypatch):
        monkeypatch.setenv("DEBUG", "1")
        import importlib

        import solvor.types

        importlib.reload(solvor.types)

        result = solvor.types.Result(
            solution=None,
            objective=0,
            iterations=100,
            status=solvor.types.Status.INFEASIBLE,
            error="constraint violated",
        )
        result.log()
        captured = capsys.readouterr()
        assert "INFEASIBLE" in captured.out
        assert "constraint violated" in captured.out

        # Cleanup
        monkeypatch.delenv("DEBUG", raising=False)
        importlib.reload(solvor.types)


class TestProgress:
    def test_progress_creation(self):
        p = Progress(iteration=10, objective=5.0, best=3.0, evaluations=100)
        assert p.iteration == 10
        assert p.objective == 5.0
        assert p.best == 3.0
        assert p.evaluations == 100

    def test_progress_defaults(self):
        p = Progress(iteration=1, objective=1.0)
        assert p.best is None
        assert p.evaluations == 0
