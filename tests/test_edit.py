from dataclasses import replace
from pathlib import Path

import yaml

from stats_projections import load_scenario
from stats_projections.config import Growth, from_dict
from stats_projections.edit import diff, to_dict

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def all_examples():
    return [load_scenario(f) for f in sorted(SCENARIOS.glob("*.yaml"))]


def test_to_dict_round_trips_every_example():
    for s in all_examples():
        assert from_dict(to_dict(s)) == s, s.name


def test_no_changes_means_empty_diff():
    s = load_scenario(SCENARIOS / "baseline.yaml")
    assert diff(s, s) == {}


def test_diff_written_as_extends_file_reproduces_edit(tmp_path):
    base = load_scenario(SCENARIOS / "baseline.yaml")
    salary = base.stream("Salary")
    edited = replace(
        base,
        starting_balance=55000,
        income=tuple(replace(x, amount=80000, growth=Growth(0.02, 0.03)) if x.name == "Salary" else x
                     for x in base.income),
        expenses=tuple(x for x in base.expenses if x.name != "Healthcare in retirement"),
        events=(*base.events[:1], replace(base.events[1], year=6), *base.events[2:]),
    )
    over = {"extends": str(SCENARIOS / "baseline.yaml"), "name": base.name, **diff(base, edited)}
    assert over["portfolio"] == {"starting_balance": 55000}
    assert {"name": "Healthcare in retirement", "remove": True} in over["expenses"]
    assert salary.amount != 80000

    f = tmp_path / "edit.yaml"
    f.write_text(yaml.safe_dump(over, sort_keys=False))
    assert replace(load_scenario(f), source=None) == replace(edited, source=None)


def test_clearing_a_field_back_to_default_is_written(tmp_path):
    base = load_scenario(SCENARIOS / "baseline.yaml")
    edited = replace(base, income=tuple(
        replace(x, loss_probability=0.0) if x.name == "Salary" else x for x in base.income))
    over = diff(base, edited)
    assert over["income"] == [{"name": "Salary", "loss_probability": 0.0}]
