"""Migrate lerobot/diffusion_pusht to the LeRobot 0.7.0 checkpoint format.

The original checkpoint predates LeRobot PR #1452 (normalization stored inside the
model weights). This script runs LeRobot's official migration unchanged, with one
workaround: the legacy config stores tuple fields (e.g. crop_shape) as JSON lists,
which 0.7.0 fails to serialize. We convert list -> tuple for tuple-typed fields,
then let the config's own validation run as usual.
"""
import dataclasses
import runpy
import sys
import types
import typing

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

REPO_ID = "lerobot/diffusion_pusht"
REVISION = "84a7c23178445c6bbf7e1a884ff497017910f653"  # pinned Hub commit
OUTPUT_DIR = "checkpoints/diffusion_pusht_migrated"


def _is_tuple_type(hint) -> bool:
    if isinstance(hint, str):  # unresolved annotation fallback
        return "tuple[" in hint
    origin = typing.get_origin(hint)
    if origin is tuple:
        return True
    if origin in (typing.Union, types.UnionType):
        return any(typing.get_origin(a) is tuple for a in typing.get_args(hint))
    return False


_original_post_init = DiffusionConfig.__post_init__


def _post_init_with_tuple_fix(self):
    try:
        hints = typing.get_type_hints(type(self))
    except Exception:
        hints = {}
    for field in dataclasses.fields(self):
        value = getattr(self, field.name)
        if isinstance(value, list) and _is_tuple_type(hints.get(field.name, field.type)):
            setattr(self, field.name, tuple(value))
            print(f"[migrate_checkpoint] {field.name}: list {value} -> tuple")
    _original_post_init(self)  # keep LeRobot's own validation


DiffusionConfig.__post_init__ = _post_init_with_tuple_fix

if __name__ == "__main__":
    sys.argv = [
        "migrate_policy_normalization",
        "--pretrained-path", REPO_ID,
        "--revision", REVISION,
        "--output-dir", OUTPUT_DIR,
        # no --push-to-hub: the migrated checkpoint stays local
    ]
    runpy.run_module("lerobot.processor.migrate_policy_normalization", run_name="__main__")