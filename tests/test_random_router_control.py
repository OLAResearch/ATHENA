import torch

from engram.tri_memory import TriMemoryAdaptor


def test_random_router_preserves_route_mass_and_is_deterministic(monkeypatch):
    adaptor = TriMemoryAdaptor(
        d_model=4,
        d_mem=3,
        hidden_size=4,
        num_layers=1,
        num_heads=2,
        adaptive_router=True,
    )
    adaptor.configure_advantage_reader(candidates="sources")
    adaptor.configure_random_router(seed=17)

    batch, time, width = 2, 3, 4
    h = torch.zeros(batch, time, width)
    mem = torch.zeros(batch, time, 3)
    source_outputs = torch.stack(
        [
            torch.full((batch, time, width), 1.0),
            torch.full((batch, time, width), 2.0),
            torch.full((batch, time, width), 3.0),
        ]
    )
    weights = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.7, 0.0, 0.3]],
            [[0.6, 0.4, 0.0], [0.5, 0.0, 0.5], [0.4, 0.3, 0.3]],
        ]
    )

    def fake_advantage_route(h, mem, cue_mem, context_h):
        del mem, cue_mem, context_h
        adaptor._last_source_outputs = source_outputs.to(h)
        adaptor._last_router_weights = weights.to(h)
        return torch.zeros_like(h), torch.zeros(1, *h.shape[:2])

    monkeypatch.setattr(adaptor, "_advantage_route", fake_advantage_route)
    adaptor.set_tri_reader_mode("random_router")
    output, _ = adaptor(h, mem)

    shuffled = adaptor.get_last_router_weights()
    assert shuffled is not None
    assert torch.equal(torch.sort(shuffled.reshape(-1))[0], torch.sort(weights.reshape(-1))[0])
    expected = sum(
        shuffled[..., index : index + 1] * source_outputs[index]
        for index in range(3)
    )
    assert torch.equal(output, expected)

    adaptor.configure_random_router(seed=17)
    output_repeat, _ = adaptor(h, mem)
    assert torch.equal(output, output_repeat)
