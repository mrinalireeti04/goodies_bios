"""Tests for task builder: ID assignment, summary, grouping."""

import pytest

from forge_trace.schemas import RefinedAction, TaskRecord, TimelineInfo
from forge_trace.task_builder import (
    assign_task_ids,
    generate_task_summary,
    _heuristic_grouping,
)


def _action(verb, start, end, obj=None, hand=None, confidence=0.8):
    return RefinedAction(
        verb=verb, object=obj, hand=hand,
        start_s=start, end_s=end, duration_s=round(end - start, 4),
        confidence=confidence, description=f"{verb} action",
    )


class TestAssignTaskIds:
    def test_chronological_ids(self, sample_refined_actions):
        tasks = assign_task_ids(sample_refined_actions)
        assert len(tasks) == 4
        assert tasks[0].task_id == "T001"
        assert tasks[1].task_id == "T002"
        assert tasks[2].task_id == "T003"
        assert tasks[3].task_id == "T004"

    def test_chronological_ordering(self):
        """Even if input is unsorted, output is ordered by start time."""
        actions = [
            _action("inspect", 10.0, 12.0),
            _action("pick up", 2.0, 4.0),
            _action("place", 6.0, 8.0),
        ]
        tasks = assign_task_ids(actions)
        assert tasks[0].verb == "pick up"
        assert tasks[1].verb == "place"
        assert tasks[2].verb == "inspect"
        assert tasks[0].task_id == "T001"

    def test_unique_ids(self):
        actions = [_action("pick up", i, i + 1) for i in range(20)]
        tasks = assign_task_ids(actions)
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_task_name_format(self):
        actions = [_action("pick up", 0, 1, obj="bolt")]
        tasks = assign_task_ids(actions)
        assert tasks[0].task_name == "Pick up bolt"

    def test_unknown_object_in_name(self):
        actions = [_action("pick up", 0, 1)]  # no object
        tasks = assign_task_ids(actions)
        assert tasks[0].task_name == "Pick up object"
        assert tasks[0].object == "unknown"

    def test_unknown_hand(self):
        actions = [_action("pick up", 0, 1)]
        tasks = assign_task_ids(actions)
        assert tasks[0].hand == "unknown"

    def test_duration_correct(self):
        actions = [_action("pick up", 2.5, 5.5)]
        tasks = assign_task_ids(actions)
        assert tasks[0].timeline.duration_s == pytest.approx(3.0, abs=0.01)

    def test_empty_input(self):
        tasks = assign_task_ids([])
        assert tasks == []


class TestGenerateTaskSummary:
    def test_summary_counts(self, sample_task_records):
        summary = generate_task_summary(sample_task_records)
        assert summary["total_tasks"] == 4
        assert summary["unique_task_types"] == 4
        assert len(summary["unique_task_list"]) == 4

    def test_repeated_tasks_counted_correctly(self):
        tasks = [
            TaskRecord(
                task_id=f"T{i:03d}", task_name="Pick up bolt", verb="pick up",
                object="bolt", hand="right",
                timeline=TimelineInfo(start_s=i * 10.0, end_s=i * 10 + 2, duration_s=2.0),
                confidence=0.8,
            )
            for i in range(1, 4)
        ]
        summary = generate_task_summary(tasks)
        assert summary["total_tasks"] == 3
        assert summary["unique_task_types"] == 1

    def test_empty(self):
        summary = generate_task_summary([])
        assert summary["total_tasks"] == 0
        assert summary["unique_task_types"] == 0


class TestHeuristicGrouping:
    def test_groups_consecutive_tasks(self):
        # 5 consecutive tasks with small gaps
        tasks = [
            TaskRecord(
                task_id=f"T{i:03d}", task_name=f"Action {i}", verb=f"action{i}",
                timeline=TimelineInfo(
                    start_s=i * 3.0, end_s=i * 3.0 + 2.0, duration_s=2.0,
                ),
                confidence=0.8,
            )
            for i in range(1, 6)
        ]
        groups = _heuristic_grouping(tasks)
        assert len(groups) >= 1
        assert len(groups[0].task_ids) >= 3

    def test_no_groups_for_sparse_tasks(self):
        tasks = [
            TaskRecord(
                task_id=f"T{i:03d}", task_name=f"Action {i}", verb=f"action{i}",
                timeline=TimelineInfo(
                    start_s=i * 30.0, end_s=i * 30.0 + 2.0, duration_s=2.0,
                ),
                confidence=0.8,
            )
            for i in range(1, 4)
        ]
        groups = _heuristic_grouping(tasks)
        assert len(groups) == 0

    def test_fewer_than_three_tasks_no_groups(self):
        tasks = [
            TaskRecord(
                task_id="T001", task_name="A", verb="a",
                timeline=TimelineInfo(start_s=0, end_s=2, duration_s=2),
                confidence=0.8,
            ),
            TaskRecord(
                task_id="T002", task_name="B", verb="b",
                timeline=TimelineInfo(start_s=3, end_s=5, duration_s=2),
                confidence=0.8,
            ),
        ]
        groups = _heuristic_grouping(tasks)
        assert len(groups) == 0
