"""
Model registry — maps model names to benchmarker classes.
"""

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a benchmarker class under a given name."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
