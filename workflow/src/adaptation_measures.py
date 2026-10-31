"""Shared helpers for the pre- and post-processing adaptation measure modules."""


def _is_current_protection(elevation) -> bool:
    """True for the {"current": True, "fill_crest": ...} form of `current_protection`.

    Current protection never clears a flood class (nor "holds" in the joint
    compound check): the baseline flood map already contains today's levees.
    """
    return isinstance(elevation, dict) and bool(elevation.get("current"))


def validate_measure_params(
    measure_type: str, params: dict, method: str, measure_def: dict
) -> None:
    """Check params against config/measures.yml bounds, type and required_for; raise ValueError."""
    param_specs = measure_def["params"]
    for name, spec in param_specs.items():
        required_for = spec.get("required_for", [])
        if method in required_for and name not in params:
            raise ValueError(
                f"Measure '{measure_type}': param '{name}' is required for method "
                f"'{method}' (config/measures.yml) but was not provided."
            )
        if (
            name in params
            and spec.get("type") == "bool"
            and not isinstance(params[name], bool)
        ):
            raise ValueError(
                f"Measure '{measure_type}': param '{name}'={params[name]!r} must be "
                f"true or false (config/measures.yml)."
            )
        if (
            name in params
            and spec.get("min") is not None
            and spec.get("max") is not None
        ):
            value = params[name]
            if isinstance(value, (int, float)) and not (
                spec["min"] <= value <= spec["max"]
            ):
                raise ValueError(
                    f"Measure '{measure_type}': param '{name}'={value} is out of bounds "
                    f"[{spec['min']}, {spec['max']}] (config/measures.yml)."
                )
