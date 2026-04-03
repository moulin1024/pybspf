from .assembly import AssemblyResult, PreprocessMeta, PreprocessedSystem, assemble_fd4_system_from_preprocessed, preprocess_fd4_system
from .cli import main
from .io import load_fields, load_mesh, parse_args
from .linear import build_amg_hierarchy, compute_operator_diagnostics, solve_with_amg_bicgstab, solve_with_existing_amg

__all__ = [
    "assemble_fd4_system_from_preprocessed",
    "AssemblyResult",
    "build_amg_hierarchy",
    "compute_operator_diagnostics",
    "load_fields",
    "load_mesh",
    "main",
    "parse_args",
    "PreprocessMeta",
    "PreprocessedSystem",
    "preprocess_fd4_system",
    "solve_with_amg_bicgstab",
    "solve_with_existing_amg",
]
