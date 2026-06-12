from lattice.config import config_preset


def test_local_contract_preset_is_cpu_safe() -> None:
    cfg = config_preset("local-contract")

    assert cfg.runtime.required_world_size == 1
    assert cfg.runtime.require_cuda is False
    assert cfg.replay.learner_batch <= 4


def test_alpha1_preset_requires_dual_gpu_ddp() -> None:
    cfg = config_preset("checkmate-alpha1")

    assert cfg.runtime.required_world_size == 2
    assert cfg.runtime.require_cuda is True
    assert cfg.runtime.require_bf16 is True
    assert cfg.runtime.require_peer_access is True
    assert cfg.replay.sampling_mode == "uniform"
    assert cfg.benchmark.movegen_path == "vectorized-tensor"
