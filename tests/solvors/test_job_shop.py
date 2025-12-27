"""Tests for Job Shop Scheduling solver."""

import pytest

from solvor.job_shop import solve_job_shop
from solvor.types import Status


class TestBasicScheduling:
    def test_single_job(self):
        """Single job with multiple operations."""
        jobs = [[(0, 3), (1, 2), (2, 1)]]  # 3 ops on machines 0, 1, 2
        result = solve_job_shop(jobs)
        assert result.objective == 6  # 3 + 2 + 1 sequential

    def test_two_jobs_no_conflict(self):
        """Two jobs on different machines."""
        jobs = [
            [(0, 3)],  # Job 0: machine 0 for 3
            [(1, 2)],  # Job 1: machine 1 for 2
        ]
        result = solve_job_shop(jobs)
        assert result.objective == 3  # Run in parallel

    def test_two_jobs_same_machine(self):
        """Two jobs competing for same machine."""
        jobs = [
            [(0, 3)],  # Job 0: machine 0 for 3
            [(0, 2)],  # Job 1: machine 0 for 2
        ]
        result = solve_job_shop(jobs)
        assert result.objective == 5  # 3 + 2 sequential

    def test_classic_3x3(self):
        """Classic 3 job, 3 machine instance."""
        jobs = [
            [(0, 3), (1, 2), (2, 2)],
            [(0, 2), (2, 1), (1, 4)],
            [(1, 4), (2, 3)],
        ]
        result = solve_job_shop(jobs)
        # Makespan depends on schedule, should be reasonable
        assert result.objective >= 7  # Lower bound
        assert result.objective <= 15  # Upper bound

    def test_empty_jobs(self):
        """Empty job list."""
        result = solve_job_shop([])
        assert result.objective == 0
        assert result.status == Status.OPTIMAL


class TestDispatchingRules:
    def test_fifo(self):
        """FIFO dispatching rule."""
        jobs = [
            [(0, 5)],
            [(0, 2)],
            [(0, 3)],
        ]
        result = solve_job_shop(jobs, rule="fifo", local_search=False)
        assert result.objective == 10  # All on same machine

    def test_spt(self):
        """Shortest Processing Time rule."""
        jobs = [
            [(0, 5)],
            [(0, 2)],
            [(0, 3)],
        ]
        result = solve_job_shop(jobs, rule="spt", local_search=False)
        assert result.objective == 10  # Same total, different order

    def test_lpt(self):
        """Longest Processing Time rule."""
        jobs = [
            [(0, 5)],
            [(0, 2)],
            [(0, 3)],
        ]
        result = solve_job_shop(jobs, rule="lpt", local_search=False)
        assert result.objective == 10

    def test_mwkr(self):
        """Most Work Remaining rule."""
        jobs = [
            [(0, 2), (1, 3)],  # Total 5
            [(0, 4)],  # Total 4
        ]
        result = solve_job_shop(jobs, rule="mwkr", local_search=False)
        assert result.objective >= 5

    def test_random(self):
        """Random dispatching rule."""
        jobs = [
            [(0, 3)],
            [(0, 2)],
        ]
        result = solve_job_shop(jobs, rule="random", local_search=False, seed=42)
        assert result.objective == 5


class TestLocalSearch:
    def test_local_search_improves(self):
        """Local search can find better solutions."""
        jobs = [
            [(0, 3), (1, 2)],
            [(1, 2), (0, 3)],
        ]
        result_no_ls = solve_job_shop(jobs, local_search=False, seed=42)
        result_ls = solve_job_shop(jobs, local_search=True, seed=42)
        # Local search should do at least as well (never worse)
        assert result_ls.objective <= result_no_ls.objective

    def test_max_iter(self):
        """Respects max_iter limit."""
        jobs = [
            [(0, 3), (1, 2)],
            [(1, 2), (0, 3)],
        ]
        result = solve_job_shop(jobs, max_iter=10, seed=42)
        assert result.iterations <= 10


class TestProgressCallback:
    def test_early_stop(self):
        """Callback can stop optimization early."""

        def callback(progress):
            return True  # Stop immediately

        jobs = [
            [(0, 3), (1, 2)],
            [(1, 2), (0, 3)],
        ]
        result = solve_job_shop(jobs, on_progress=callback, progress_interval=1, seed=42)
        assert result.status == Status.FEASIBLE


class TestValidation:
    def test_empty_job(self):
        """Job with no operations raises error."""
        with pytest.raises(ValueError, match="has no operations"):
            solve_job_shop([[(0, 3)], []])

    def test_negative_machine(self):
        """Negative machine index raises error."""
        with pytest.raises(ValueError, match="negative machine index"):
            solve_job_shop([[(-1, 3)]])

    def test_negative_duration(self):
        """Negative duration raises error."""
        with pytest.raises(ValueError, match="negative duration"):
            solve_job_shop([[(0, -3)]])

    def test_unknown_rule(self):
        """Unknown dispatching rule raises error."""
        with pytest.raises(ValueError, match="Unknown dispatching rule"):
            solve_job_shop([[(0, 3)]], rule="unknown")


class TestScheduleValidity:
    def test_job_precedence(self):
        """Operations within a job are in order."""
        jobs = [
            [(0, 3), (1, 2), (2, 1)],
        ]
        result = solve_job_shop(jobs)
        schedule = result.solution

        # Check precedence
        for j, job in enumerate(jobs):
            for op_idx in range(1, len(job)):
                prev_end = schedule[(j, op_idx - 1)][1]
                curr_start = schedule[(j, op_idx)][0]
                assert curr_start >= prev_end

    def test_machine_no_overlap(self):
        """No two operations overlap on same machine."""
        jobs = [
            [(0, 3), (1, 2)],
            [(0, 2), (1, 3)],
        ]
        result = solve_job_shop(jobs)
        schedule = result.solution

        # Group by machine
        by_machine: dict[int, list] = {}
        for j, job in enumerate(jobs):
            for op_idx, (machine, _) in enumerate(job):
                if machine not in by_machine:
                    by_machine[machine] = []
                by_machine[machine].append((j, op_idx))

        # Check no overlap on each machine
        for machine, ops in by_machine.items():
            intervals = [(schedule[op][0], schedule[op][1]) for op in ops]
            intervals.sort()
            for i in range(1, len(intervals)):
                assert intervals[i][0] >= intervals[i - 1][1]


class TestLargerInstances:
    def test_5_jobs(self):
        """5 job instance."""
        jobs = [
            [(0, 2), (1, 3)],
            [(1, 4), (0, 2)],
            [(0, 3), (1, 1)],
            [(1, 2), (0, 4)],
            [(0, 1), (1, 2)],
        ]
        result = solve_job_shop(jobs, seed=42)
        # Should find a reasonable solution
        assert result.objective <= 20

    def test_many_machines(self):
        """Jobs across many machines."""
        jobs = [
            [(i, 1) for i in range(10)],  # Job 0: 10 ops
            [(i, 1) for i in range(10)],  # Job 1: 10 ops
        ]
        result = solve_job_shop(jobs, seed=42)
        # Minimum possible is 10 (each op takes 1)
        assert result.objective >= 10
