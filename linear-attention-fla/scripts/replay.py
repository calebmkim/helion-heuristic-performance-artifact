"""Record and explicitly replay constituent Helion kernel configurations."""

from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any
from typing import Iterator


def _value_key(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "tensor": {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        }
    if value is None or type(value) in (bool, int, float, str):
        return {"scalar": [type(value).__name__, value]}
    if isinstance(value, (tuple, list)):
        return {
            type(value).__name__: [_value_key(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "dict": [
                [str(key), _value_key(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return {type(value).__name__: str(value)}
    if type(value).__name__ == "ConstExpr" and hasattr(value, "value"):
        return {"constexpr": _value_key(value.value)}
    raise TypeError(f"cannot make a replay key for {type(value).__name__}")


def argument_key(args: tuple[object, ...]) -> list[object]:
    return [_value_key(value) for value in args]


def _key_text(key: object) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _engine_kernels() -> tuple[Any, dict[str, Any]]:
    from examples.linear import linear_attention_engine as engine
    from helion.runtime.kernel import Kernel

    return engine, {
        name: value
        for name, value in vars(engine).items()
        if isinstance(value, Kernel)
    }


class ConfigRecorder:
    """Patch engine dispatch and record the config selected for every call."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.engine, self.kernels = _engine_kernels()
        self._calls: dict[tuple[str, str], dict[str, Any]] = {}
        self._order: list[tuple[str, str]] = []
        self._sequence: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._calls.clear()
        self._order.clear()
        self._sequence.clear()

    def _selected_by_aot(self, kernel_name: str, config: dict[str, object]) -> bool:
        if self.arm != "aot_tuned":
            return False
        from helion.autotuner.aot_cache import AOTAutotuneCache

        return any(
            candidate_name == kernel_name and dict(candidate) == config
            for (_source, candidate_name, _shape), candidate in (
                AOTAutotuneCache._heuristic_results.items()
            )
        )

    def _wrapper(self, name: str, kernel: Any) -> Any:
        def invoke(*args: object, **kwargs: object) -> Any:
            normalized = kernel.normalize_args(*args, **kwargs)
            result = kernel(*normalized)
            bound = kernel.bind(normalized)
            if bound._config is None:
                raise RuntimeError(f"{name} executed without selecting a config")
            config = dict(bound._config)
            key = argument_key(normalized)
            record_key = (name, _key_text(key))
            self._sequence.append(record_key)
            existing = self._calls.get(record_key)
            if existing is None:
                spec = bound.config_spec
                existing = {
                    "kernel": name,
                    "call_key": key,
                    "config": config,
                    "count": 0,
                    "selection": {
                        "heuristics_fired": list(spec.autotuner_heuristics),
                        "selected_by_aot_cache": self._selected_by_aot(
                            kernel.name, config
                        ),
                        "matmul_facts": [
                            [fact.static_m, fact.static_n, fact.static_k]
                            for fact in spec.matmul_facts
                        ],
                    },
                }
                self._calls[record_key] = existing
                self._order.append(record_key)
            elif existing["config"] != config:
                raise RuntimeError(
                    f"{name} selected multiple configs for the same replay key"
                )
            existing["count"] += 1
            return result

        invoke.__name__ = name
        return invoke

    @contextmanager
    def patched(self) -> Iterator[None]:
        for name, kernel in self.kernels.items():
            setattr(self.engine, name, self._wrapper(name, kernel))
        try:
            yield
        finally:
            for name, kernel in self.kernels.items():
                setattr(self.engine, name, kernel)

    def plan(self) -> dict[str, object]:
        calls = [self._calls[key] for key in self._order]
        if not calls:
            raise RuntimeError("the workload did not invoke a Helion engine kernel")
        if self.arm == "aot_tuned":
            missed = [
                call["kernel"]
                for call in calls
                if not call["selection"]["selected_by_aot_cache"]
            ]
            if missed:
                raise RuntimeError(
                    "AOT selector did not supply configs for: "
                    + ", ".join(sorted(set(missed)))
                )
        call_indices = {key: index for index, key in enumerate(self._order)}
        return {
            "calls": calls,
            "sequence": [call_indices[key] for key in self._sequence],
        }


class ConfigReplay:
    """Replay materialized configs while keeping all arms in one process."""

    def __init__(self, arm_plans: dict[str, dict[str, object]]) -> None:
        self.engine, self.kernels = _engine_kernels()
        self._plans: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        self._sequences: dict[
            str, list[tuple[str, str, dict[str, Any]]]
        ] = {}
        for arm, plan in arm_plans.items():
            entries: dict[tuple[str, str], dict[str, Any]] = {}
            ordered_entries: list[tuple[str, str, dict[str, Any]]] = []
            for call in plan["calls"]:  # type: ignore[index]
                key = (call["kernel"], _key_text(call["call_key"]))
                if key in entries:
                    raise ValueError(f"duplicate replay entry for {arm}: {key[0]}")
                entries[key] = call
                ordered_entries.append((key[0], key[1], call))
            self._plans[arm] = entries
            self._sequences[arm] = [
                ordered_entries[index]
                for index in plan["sequence"]  # type: ignore[index]
            ]
        self._active_arm: str | None = None
        self._audit = False
        self._position = 0
        self._compiled: dict[tuple[str, str, str], Any] = {}
        self._trailing_defaults: dict[tuple[str, int], tuple[object, ...]] = {}

    def _wrapper(self, name: str, kernel: Any) -> Any:
        def invoke(*args: object, **kwargs: object) -> Any:
            if self._active_arm is None:
                raise RuntimeError(f"{name} was called without an active replay arm")
            sequence = self._sequences[self._active_arm]
            if self._position >= len(sequence):
                raise RuntimeError(
                    f"extra {name} call in {self._active_arm} replay"
                )
            sequence_position = self._position
            expected_name, expected_key, entry = sequence[sequence_position]
            self._position += 1
            if name != expected_name:
                raise RuntimeError(
                    f"{self._active_arm} replay expected {expected_name}, got {name}"
                )
            cache_key = (self._active_arm, expected_name, expected_key)
            compiled = self._compiled.get(cache_key)
            if compiled is None or self._audit:
                normalized = kernel.normalize_args(*args, **kwargs)
                observed_key = _key_text(argument_key(normalized))
                if observed_key != expected_key:
                    raise RuntimeError(
                        f"{self._active_arm} replay arguments changed for {name}"
                    )
                if not kwargs:
                    self._trailing_defaults[
                        (self._active_arm, sequence_position)
                    ] = normalized[len(args) :]
            elif kwargs:
                normalized = kernel.normalize_args(*args, **kwargs)
            else:
                normalized = (
                    *args,
                    *self._trailing_defaults[
                        (self._active_arm, sequence_position)
                    ],
                )
            if compiled is not None:
                return compiled(*normalized)
            from helion.runtime.config import Config

            compiled = kernel.bind(normalized).compile_config(
                Config.from_dict(entry["config"])
            )
            self._compiled[cache_key] = compiled
            return compiled(*normalized)

        invoke.__name__ = name
        return invoke

    @contextmanager
    def patched(self) -> Iterator[None]:
        for name, kernel in self.kernels.items():
            setattr(self.engine, name, self._wrapper(name, kernel))
        try:
            yield
        finally:
            for name, kernel in self.kernels.items():
                setattr(self.engine, name, kernel)

    @contextmanager
    def use(self, arm: str, *, audit: bool = False) -> Iterator[None]:
        if arm not in self._plans:
            raise KeyError(f"no replay plan for {arm}")
        if self._active_arm is not None:
            raise RuntimeError("replay arm contexts cannot be nested")
        self._active_arm = arm
        self._audit = audit
        self._position = 0
        try:
            yield
            expected = len(self._sequences[arm])
            if self._position != expected:
                raise RuntimeError(
                    f"{arm} replay used {self._position}/{expected} planned calls"
                )
        finally:
            self._active_arm = None
            self._audit = False
            self._position = 0
