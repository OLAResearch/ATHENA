import torch
import pytest

from scripts.routing_proportion import RoutingProportionCollector


class _Adaptor:
    def __init__(self, weights):
        self.weights = weights

    def get_last_router_weights(self):
        return self.weights


class _Wrapper:
    def __init__(self, weights):
        self.adaptor = _Adaptor(weights)


def test_route_report_distinguishes_soft_mass_and_argmax_winners():
    weights = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.2, 0.0, 0.8]]]
    )
    wrapper = _Wrapper(weights)
    collector = RoutingProportionCollector()
    collector.add_last_token(wrapper)
    report = collector.report()

    assert report["positions"] == 1
    assert report["weighted_source_mass"]["GH"] == pytest.approx(0.8)
    assert report["argmax_source_fraction"] == {"E": 0.0, "GE": 0.0, "GH": 1.0}
    assert report["generated_admission_rate"] == 1.0


def test_route_report_counts_exact_e_fallback_and_generated_winners():
    weights = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.5, 0.0, 0.5]]]
    )
    collector = RoutingProportionCollector()
    collector.add_weights(weights)
    report = collector.report()

    assert report["positions"] == 3
    assert report["exact_e_fallback_rate"] == pytest.approx(1 / 3)
    assert report["generated_admission_rate"] == pytest.approx(2 / 3)
    assert report["generated_argmax_counts"] == {"GE": 1, "GH": 1}
