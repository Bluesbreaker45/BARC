#!/usr/bin/env python3
"""Deterministic AST-based call-site inliner for common.py helpers.

This script expands calls to selected helper functions directly at call sites.
Supported call contexts (phase-1):
- standalone expression call, e.g. foo(...)
- assignment RHS call, e.g. x = foo(...)

Unsupported contexts are skipped and logged.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass
class InlineStats:
    target: str
    total_calls: int = 0
    supported_context_calls: int = 0
    transformed_calls: int = 0
    simply_transformer_calls: int = 0
    no_prefix_inlines: int = 0
    prefixed_inlines: int = 0
    skipped_unsupported_context_calls: int = 0
    skipped_shadowed_calls: int = 0
    skipped_signature_calls: int = 0
    skipped_calls: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    prefixed_reasons: dict[str, int] = field(default_factory=dict)
    signature_error_reasons: dict[str, int] = field(default_factory=dict)
    context_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "total_calls": self.total_calls,
            "supported_context_calls": self.supported_context_calls,
            "transformed_calls": self.transformed_calls,
            "simply_transformer_calls": self.simply_transformer_calls,
            "no_prefix_inlines": self.no_prefix_inlines,
            "prefixed_inlines": self.prefixed_inlines,
            "skipped_unsupported_context_calls": self.skipped_unsupported_context_calls,
            "skipped_shadowed_calls": self.skipped_shadowed_calls,
            "skipped_signature_calls": self.skipped_signature_calls,
            "skipped_calls": self.skipped_calls,
            "skipped_reasons": self.skipped_reasons,
            "prefixed_reasons": self.prefixed_reasons,
            "signature_error_reasons": self.signature_error_reasons,
            "context_counts": self.context_counts,
        }


@dataclass
class InlineBuildResult:
    block: list[ast.stmt]
    used_prefix: bool
    prefix_reason: str | None


class CallCounter(ast.NodeVisitor):
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.count = 0

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == self.target_name:
            self.count += 1
        self.generic_visit(node)


class CallContextCounter(ast.NodeVisitor):
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.counts: Counter[str] = Counter()
        self._parents: list[ast.AST] = []

    def visit(self, node: ast.AST) -> Any:
        self._parents.append(node)
        result = super().visit(node)
        self._parents.pop()
        return result

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == self.target_name:
            parent = self._parents[-2] if len(self._parents) >= 2 else None
            if isinstance(parent, ast.Expr) and parent.value is node:
                self.counts["supported-expr"] += 1
            elif isinstance(parent, ast.Assign) and parent.value is node:
                self.counts["supported-assign"] += 1
            elif isinstance(parent, ast.AnnAssign) and parent.value is node:
                self.counts["supported-annassign"] += 1
            else:
                self.counts["unsupported-context"] += 1
        self.generic_visit(node)


class ReadNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.target)
        self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)


class ScopedNameRenamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            new = copy.deepcopy(node)
            new.arg = self.mapping[node.arg]
            return new
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new = copy.deepcopy(node)
        if new.name in self.mapping:
            new.name = self.mapping[new.name]
        return new

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        new = copy.deepcopy(node)
        if new.name in self.mapping:
            new.name = self.mapping[new.name]
        return new

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        new = copy.deepcopy(node)
        if new.name in self.mapping:
            new.name = self.mapping[new.name]
        return new


class ReturnRewriter(ast.NodeTransformer):
    def __init__(self, assign_targets: list[ast.expr] | None, done_var: str | None) -> None:
        self.assign_targets = assign_targets
        self.done_var = done_var

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        value = copy.deepcopy(node.value) if node.value is not None else ast.Constant(value=None)

        rewritten: list[ast.stmt] = []
        if self.assign_targets is not None:
            rewritten.append(
                ast.Assign(targets=[copy.deepcopy(t) for t in self.assign_targets], value=value)
            )
        elif node.value is not None:
            # Preserve side effects in return expressions for expression-statement calls.
            rewritten.append(ast.Expr(value=value))

        if self.done_var is not None:
            rewritten.append(
                ast.Assign(targets=[ast.Name(id=self.done_var, ctx=ast.Store())], value=ast.Constant(value=True))
            )
        return rewritten

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


@dataclass
class FunctionSpec:
    name: str
    def_node: ast.FunctionDef

    @property
    def posonly(self) -> list[ast.arg]:
        return list(self.def_node.args.posonlyargs)

    @property
    def pos_or_kw(self) -> list[ast.arg]:
        return list(self.def_node.args.args)

    @property
    def kwonly(self) -> list[ast.arg]:
        return list(self.def_node.args.kwonlyargs)

    @property
    def defaults(self) -> list[ast.expr]:
        return list(self.def_node.args.defaults)

    @property
    def kw_defaults(self) -> list[ast.expr | None]:
        return list(self.def_node.args.kw_defaults)

    @property
    def has_vararg(self) -> bool:
        return self.def_node.args.vararg is not None

    @property
    def has_kwarg(self) -> bool:
        return self.def_node.args.kwarg is not None


class DeterministicInliner:
    def __init__(self, common_file: Path, target_names: list[str]) -> None:
        self.common_file = common_file
        self.target_names = target_names
        self.function_specs = self._load_function_specs(common_file, target_names)
        self._inline_counter = 0

    @staticmethod
    def _load_function_specs(common_file: Path, target_names: list[str]) -> dict[str, FunctionSpec]:
        src = common_file.read_text(encoding="utf-8")
        tree = ast.parse(src)
        defs: dict[str, FunctionSpec] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in target_names:
                defs[node.name] = FunctionSpec(name=node.name, def_node=node)

        missing = [name for name in target_names if name not in defs]
        if missing:
            raise ValueError(f"Missing function definitions in common.py: {missing}")
        return defs

    def inline_target(self, code_str: str, target_name: str) -> tuple[str, InlineStats]:
        spec = self.function_specs[target_name]
        source_tree = ast.parse(code_str)

        counter = CallCounter(target_name)
        counter.visit(source_tree)

        context_counter = CallContextCounter(target_name)
        context_counter.visit(source_tree)

        supported_context_calls = (
            context_counter.counts.get("supported-expr", 0)
            + context_counter.counts.get("supported-assign", 0)
            + context_counter.counts.get("supported-annassign", 0)
        )
        unsupported_context_calls = context_counter.counts.get("unsupported-context", 0)

        stats = InlineStats(
            target=target_name,
            total_calls=counter.count,
            supported_context_calls=supported_context_calls,
            skipped_unsupported_context_calls=unsupported_context_calls,
            context_counts=dict(context_counter.counts),
        )
        transformed_tree = self._transform_tree(source_tree, spec, stats)

        stats.skipped_calls = max(0, stats.total_calls - stats.transformed_calls)
        skipped_known = (
            stats.skipped_unsupported_context_calls
            + stats.skipped_shadowed_calls
            + stats.skipped_signature_calls
        )

        if stats.skipped_unsupported_context_calls > 0:
            stats.skipped_reasons["unsupported-context"] = stats.skipped_unsupported_context_calls
        if stats.skipped_shadowed_calls > 0:
            stats.skipped_reasons["shadowed-target-name"] = stats.skipped_shadowed_calls
        if stats.skipped_signature_calls > 0:
            stats.skipped_reasons["unsupported-signature-or-call-shape"] = stats.skipped_signature_calls

        remaining_unclassified = max(0, stats.skipped_calls - skipped_known)
        if remaining_unclassified > 0:
            stats.skipped_reasons["unclassified"] = remaining_unclassified

        ast.fix_missing_locations(transformed_tree)
        rendered = self._render_with_preserved_comments(
            original_code=code_str,
            original_tree=source_tree,
            transformed_tree=transformed_tree,
        )
        return rendered, stats

    def _render_with_preserved_comments(
        self,
        original_code: str,
        original_tree: ast.Module,
        transformed_tree: ast.Module,
    ) -> str:
        old_body = original_tree.body
        new_body = transformed_tree.body
        if len(old_body) != len(new_body):
            return ast.unparse(transformed_tree) + "\n"

        original_lines = original_code.splitlines(keepends=True)
        rendered_parts: list[str] = []
        cursor = 1

        for old_node, new_node in zip(old_body, new_body):
            if old_node.lineno is None or old_node.end_lineno is None:
                return ast.unparse(transformed_tree) + "\n"

            old_start = old_node.lineno
            old_end = old_node.end_lineno
            rendered_parts.extend(original_lines[cursor - 1 : old_start - 1])

            if self._ast_equal_without_locations(old_node, new_node):
                rendered_parts.extend(original_lines[old_start - 1 : old_end])
            else:
                rendered_parts.append(self._render_node_with_preserved_comments(old_node, new_node, original_lines))
                rendered_parts.append("\n")

            cursor = old_end + 1

        rendered_parts.extend(original_lines[cursor - 1 :])
        rendered = "".join(rendered_parts)
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def _render_node_with_preserved_comments(
        self,
        old_node: ast.stmt,
        new_node: ast.stmt,
        original_lines: list[str],
    ) -> str:
        rendered = ast.unparse(new_node)

        if old_node.lineno is None or old_node.end_lineno is None:
            return rendered

        old_segment = [line.rstrip("\n") for line in original_lines[old_node.lineno - 1 : old_node.end_lineno]]
        old_comment_entries = [
            (idx, line)
            for idx, line in enumerate(old_segment)
            if line.lstrip().startswith("#")
        ]
        if not old_comment_entries:
            return rendered

        rendered_lines = rendered.splitlines()
        existing_comment_counts: Counter[str] = Counter(
            line.strip()
            for line in rendered_lines
            if line.strip().startswith("#")
        )

        missing_comments: list[tuple[int, str]] = []
        for idx, comment_line in old_comment_entries:
            key = comment_line.strip()
            if existing_comment_counts.get(key, 0) > 0:
                existing_comment_counts[key] -= 1
            else:
                missing_comments.append((idx, comment_line))

        if not missing_comments:
            return rendered

        for old_idx, comment_line in missing_comments:
            insert_idx = self._choose_comment_insertion_index(
                old_segment=old_segment,
                rendered_lines=rendered_lines,
                old_comment_idx=old_idx,
                is_compound=isinstance(new_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)),
            )
            rendered_lines.insert(insert_idx, comment_line)

        return "\n".join(rendered_lines)

    def _choose_comment_insertion_index(
        self,
        old_segment: list[str],
        rendered_lines: list[str],
        old_comment_idx: int,
        is_compound: bool,
    ) -> int:
        next_anchor = self._find_old_neighbor_code_line(old_segment, old_comment_idx, search_next=True)
        if next_anchor is not None:
            anchor_idx = self._find_rendered_code_line(rendered_lines, next_anchor, from_end=False)
            if anchor_idx is not None:
                return anchor_idx

        prev_anchor = self._find_old_neighbor_code_line(old_segment, old_comment_idx, search_next=False)
        if prev_anchor is not None:
            anchor_idx = self._find_rendered_code_line(rendered_lines, prev_anchor, from_end=True)
            if anchor_idx is not None:
                return anchor_idx + 1

        if is_compound and len(rendered_lines) > 1:
            return 1
        return 0

    def _find_old_neighbor_code_line(
        self,
        lines: list[str],
        pivot: int,
        search_next: bool,
    ) -> str | None:
        if search_next:
            idx_iter = range(pivot + 1, len(lines))
        else:
            idx_iter = range(pivot - 1, -1, -1)

        for idx in idx_iter:
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped
        return None

    def _find_rendered_code_line(
        self,
        lines: list[str],
        target_stripped: str,
        from_end: bool,
    ) -> int | None:
        if from_end:
            idx_iter = range(len(lines) - 1, -1, -1)
        else:
            idx_iter = range(len(lines))

        for idx in idx_iter:
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == target_stripped:
                return idx
        return None

    def _ast_equal_without_locations(self, lhs: ast.AST, rhs: ast.AST) -> bool:
        return ast.dump(lhs, include_attributes=False) == ast.dump(rhs, include_attributes=False)

    def _transform_tree(self, tree: ast.Module, spec: FunctionSpec, stats: InlineStats) -> ast.Module:
        tree_copy = copy.deepcopy(tree)
        module_shadowed = self._collect_scope_names(tree_copy.body)
        tree_copy.body = self._transform_block(tree_copy.body, spec, stats, module_shadowed)
        return tree_copy

    def _collect_scope_names(self, stmts: list[ast.stmt]) -> set[str]:
        collector = LocalNameCollector()
        for stmt in stmts:
            collector.visit(stmt)
        return collector.names

    def _function_scope_names(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        names = self._collect_scope_names(func.body)
        names.update(arg.arg for arg in func.args.posonlyargs)
        names.update(arg.arg for arg in func.args.args)
        names.update(arg.arg for arg in func.args.kwonlyargs)
        if func.args.vararg is not None:
            names.add(func.args.vararg.arg)
        if func.args.kwarg is not None:
            names.add(func.args.kwarg.arg)
        return names

    def _transform_block(
        self,
        stmts: list[ast.stmt],
        spec: FunctionSpec,
        stats: InlineStats,
        shadowed_names: set[str],
    ) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for idx, stmt in enumerate(stmts):
            following_stmts = stmts[idx + 1 :]
            replaced = self._try_inline_statement(
                stmt,
                spec,
                shadowed_names,
                stats,
                following_stmts=following_stmts,
            )
            if replaced is not None:
                out.extend(replaced.block)
                stats.transformed_calls += 1
                if replaced.used_prefix:
                    stats.prefixed_inlines += 1
                    if replaced.prefix_reason is not None:
                        stats.prefixed_reasons[replaced.prefix_reason] = (
                            stats.prefixed_reasons.get(replaced.prefix_reason, 0) + 1
                        )
                else:
                    stats.no_prefix_inlines += 1
                    stats.simply_transformer_calls += 1
                continue

            out.append(self._recurse_statement(stmt, spec, stats, shadowed_names))
        return out

    def _recurse_statement(
        self,
        stmt: ast.stmt,
        spec: FunctionSpec,
        stats: InlineStats,
        shadowed_names: set[str],
    ) -> ast.stmt:
        node = copy.deepcopy(stmt)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_shadowed = set(shadowed_names) | self._function_scope_names(node)
            node.body = self._transform_block(node.body, spec, stats, local_shadowed)
            return node
        if isinstance(node, ast.ClassDef):
            node.body = self._transform_block(node.body, spec, stats, set(shadowed_names))
            return node
        if isinstance(node, ast.If):
            node.body = self._transform_block(node.body, spec, stats, shadowed_names)
            node.orelse = self._transform_block(node.orelse, spec, stats, shadowed_names)
            return node
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            node.body = self._transform_block(node.body, spec, stats, shadowed_names)
            node.orelse = self._transform_block(node.orelse, spec, stats, shadowed_names)
            return node
        if isinstance(node, ast.With):
            node.body = self._transform_block(node.body, spec, stats, shadowed_names)
            return node
        if isinstance(node, ast.AsyncWith):
            node.body = self._transform_block(node.body, spec, stats, shadowed_names)
            return node
        if isinstance(node, ast.Try):
            node.body = self._transform_block(node.body, spec, stats, shadowed_names)
            node.orelse = self._transform_block(node.orelse, spec, stats, shadowed_names)
            node.finalbody = self._transform_block(node.finalbody, spec, stats, shadowed_names)
            for handler in node.handlers:
                handler.body = self._transform_block(handler.body, spec, stats, shadowed_names)
            return node
        return node

    def _extract_supported_call(
        self,
        stmt: ast.stmt,
        target_name: str,
    ) -> tuple[ast.Call, list[ast.expr] | None] | None:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == target_name:
                return call, None

        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == target_name:
                return call, stmt.targets

        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == target_name:
                return call, [stmt.target]

        return None

    def _try_inline_statement(
        self,
        stmt: ast.stmt,
        spec: FunctionSpec,
        shadowed_names: set[str],
        stats: InlineStats,
        following_stmts: list[ast.stmt],
    ) -> InlineBuildResult | None:
        extracted = self._extract_supported_call(stmt, spec.name)
        if spec.name in shadowed_names:
            if extracted is not None:
                stats.skipped_shadowed_calls += 1
            return None

        if extracted is None:
            return None

        call, assign_targets = extracted
        result, error_reason = self._build_inline_block(
            call,
            spec,
            assign_targets=assign_targets,
            shadowed_names=shadowed_names,
            following_stmts=following_stmts,
        )
        if error_reason is not None:
            stats.skipped_signature_calls += 1
            stats.signature_error_reasons[error_reason] = (
                stats.signature_error_reasons.get(error_reason, 0) + 1
            )
            return None
        return result

    def _build_inline_block(
        self,
        call: ast.Call,
        spec: FunctionSpec,
        assign_targets: list[ast.expr] | None,
        shadowed_names: set[str],
        following_stmts: list[ast.stmt],
    ) -> tuple[InlineBuildResult | None, str | None]:
        self._inline_counter += 1
        prefix = f"_inl_{spec.name}_{self._inline_counter}"
        done_var = f"{prefix}_done"

        bound_values, bind_error = self._bind_call_arguments(call, spec)
        if bind_error is not None:
            return None, bind_error
        if bound_values is None:
            return None, "unknown signature binding error"

        body_clone = [copy.deepcopy(stmt) for stmt in spec.def_node.body]
        if (
            body_clone
            and isinstance(body_clone[0], ast.Expr)
            and isinstance(body_clone[0].value, ast.Constant)
            and isinstance(body_clone[0].value.value, str)
        ):
            body_clone = body_clone[1:]

        local_collector = LocalNameCollector()
        for stmt in body_clone:
            local_collector.visit(stmt)

        param_names = [arg.arg for arg in (spec.posonly + spec.pos_or_kw + spec.kwonly)]
        param_name_set = set(param_names)
        colliding_params = param_name_set & shadowed_names
        later_reads = self._collect_read_names_current_scope(following_stmts)
        params_requiring_restore = {name for name in colliding_params if name in later_reads}

        prefixable_local_names = set(local_collector.names)
        has_any_return = self._contains_return_current_scope(body_clone)
        requires_done_guard = has_any_return and self._has_early_return(spec.def_node)
        track_return_for_assignment = assign_targets is not None and has_any_return
        return_guaranteed = track_return_for_assignment and self._all_paths_return_current_scope(spec.def_node.body)
        needs_done_flag = requires_done_guard or (track_return_for_assignment and not return_guaranteed)
        colliding_local_names = prefixable_local_names & shadowed_names

        safe_overwrite_names: set[str] = set()
        if assign_targets is not None:
            assigned_names = self._extract_assigned_names(assign_targets)
            safe_overwrite_names = colliding_local_names & assigned_names

        names_to_prefix = colliding_local_names - safe_overwrite_names
        use_prefix_for_locals = len(names_to_prefix) > 0
        prefix_reason: str | None = "name-collision" if use_prefix_for_locals else None
        rename_map = {name: f"{prefix}_{name}" for name in sorted(names_to_prefix)}

        renamer = ScopedNameRenamer(rename_map)
        renamed_body = [renamer.visit(stmt) for stmt in body_clone]

        rewritten_body: list[ast.stmt] = []
        if has_any_return:
            rewriter = ReturnRewriter(
                assign_targets=assign_targets,
                done_var=done_var if needs_done_flag else None,
            )
            for stmt in renamed_body:
                replaced = rewriter.visit(stmt)
                if isinstance(replaced, list):
                    rewritten_body.extend(replaced)
                elif isinstance(replaced, ast.stmt):
                    rewritten_body.append(replaced)
                else:
                    return None, "unexpected return rewrite shape"
        else:
            rewritten_body = renamed_body

        if requires_done_guard:
            final_body = self._guard_statements(rewritten_body, done_var)
        else:
            final_body = rewritten_body

        param_assignments: list[ast.stmt] = []
        for param in param_names:
            target_name = rename_map.get(param, param)
            param_assignments.append(
                ast.Assign(
                    targets=[ast.Name(id=target_name, ctx=ast.Store())],
                    value=copy.deepcopy(bound_values[param]),
                )
            )

        param_backup_stmts: list[ast.stmt] = []
        param_restore_stmts: list[ast.stmt] = []
        for param in sorted(params_requiring_restore):
            saved_name = f"{prefix}_saved_{param}"
            param_backup_stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=saved_name, ctx=ast.Store())],
                    value=ast.Name(id=param, ctx=ast.Load()),
                )
            )
            param_restore_stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=ast.Name(id=saved_name, ctx=ast.Load()),
                )
            )

        core_block: list[ast.stmt] = []
        core_block.extend(param_assignments)
        if needs_done_flag:
            core_block.append(ast.Assign(targets=[ast.Name(id=done_var, ctx=ast.Store())], value=ast.Constant(False)))
        core_block.extend(final_body)
        if assign_targets is not None:
            if has_any_return and needs_done_flag:
                core_block.append(
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=done_var, ctx=ast.Load())),
                        body=[ast.Assign(targets=[copy.deepcopy(t) for t in assign_targets], value=ast.Constant(None))],
                        orelse=[],
                    )
                )
            elif not has_any_return:
                core_block.append(
                    ast.Assign(targets=[copy.deepcopy(t) for t in assign_targets], value=ast.Constant(None))
                )
            else:
                # All control-flow paths in callee return and write assign_targets directly.
                pass

        inline_block: list[ast.stmt] = []
        if param_backup_stmts:
            inline_block.extend(param_backup_stmts)
            inline_block.extend(core_block)
            inline_block.extend(param_restore_stmts)
        else:
            inline_block.extend(core_block)

        return (
            InlineBuildResult(
                block=inline_block,
                used_prefix=use_prefix_for_locals,
                prefix_reason=prefix_reason,
            ),
            None,
        )

    def _collect_read_names_current_scope(self, stmts: list[ast.stmt]) -> set[str]:
        collector = ReadNameCollector()
        for stmt in stmts:
            collector.visit(stmt)
        return collector.reads

    def _extract_assigned_names(self, targets: list[ast.expr]) -> set[str]:
        names: set[str] = set()

        def _visit_target(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.expr):
                        _visit_target(elt)
                return
            if isinstance(target, ast.Starred) and isinstance(target.value, ast.expr):
                _visit_target(target.value)

        for target in targets:
            _visit_target(target)
        return names

    def _all_paths_return_current_scope(self, stmts: list[ast.stmt]) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                return True
            if isinstance(stmt, ast.If):
                if self._all_paths_return_current_scope(stmt.body) and self._all_paths_return_current_scope(stmt.orelse):
                    return True
                continue
        return False

    def _contains_return_current_scope(self, stmts: list[ast.stmt]) -> bool:
        class _ReturnFinder(ast.NodeVisitor):
            def __init__(self) -> None:
                self.found = False

            def visit_Return(self, node: ast.Return) -> None:
                self.found = True

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        finder = _ReturnFinder()
        for stmt in stmts:
            if finder.found:
                break
            finder.visit(stmt)
        return finder.found

    def _has_early_return(self, func_def: ast.FunctionDef) -> bool:
        return self._block_has_early_return(func_def.body, is_tail_position=True)

    def _block_has_early_return(self, stmts: list[ast.stmt], is_tail_position: bool) -> bool:
        for idx, stmt in enumerate(stmts):
            is_last_stmt = idx == len(stmts) - 1
            stmt_is_tail = is_tail_position and is_last_stmt

            if isinstance(stmt, ast.Return):
                if not stmt_is_tail:
                    return True
                continue

            if isinstance(stmt, ast.If):
                branch_tail = stmt_is_tail
                if self._block_has_early_return(stmt.body, is_tail_position=branch_tail):
                    return True
                if self._block_has_early_return(stmt.orelse, is_tail_position=branch_tail):
                    return True
                continue

            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                if self._block_has_early_return(stmt.body, is_tail_position=stmt_is_tail):
                    return True
                continue

            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                if self._contains_return_current_scope([stmt]):
                    return True

        return False

    def _guard_statements(self, stmts: list[ast.stmt], done_var: str) -> list[ast.stmt]:
        guarded: list[ast.stmt] = []
        for stmt in stmts:
            stmt_copy = copy.deepcopy(stmt)
            stmt_copy = self._guard_compound(stmt_copy, done_var)
            guarded.append(
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=done_var, ctx=ast.Load())),
                    body=[stmt_copy],
                    orelse=[],
                )
            )
        return guarded

    def _guard_compound(self, stmt: ast.stmt, done_var: str) -> ast.stmt:
        if isinstance(stmt, ast.If):
            stmt.body = self._guard_statements(stmt.body, done_var)
            stmt.orelse = self._guard_statements(stmt.orelse, done_var)
            return stmt
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            stmt.body = self._guard_statements(stmt.body, done_var)
            stmt.body.append(
                ast.If(
                    test=ast.Name(id=done_var, ctx=ast.Load()),
                    body=[ast.Break()],
                    orelse=[],
                )
            )
            stmt.orelse = self._guard_statements(stmt.orelse, done_var)
            return stmt
        if isinstance(stmt, ast.With):
            stmt.body = self._guard_statements(stmt.body, done_var)
            return stmt
        if isinstance(stmt, ast.AsyncWith):
            stmt.body = self._guard_statements(stmt.body, done_var)
            return stmt
        if isinstance(stmt, ast.Try):
            stmt.body = self._guard_statements(stmt.body, done_var)
            stmt.orelse = self._guard_statements(stmt.orelse, done_var)
            stmt.finalbody = self._guard_statements(stmt.finalbody, done_var)
            for handler in stmt.handlers:
                handler.body = self._guard_statements(handler.body, done_var)
            return stmt
        return stmt

    def _bind_call_arguments(
        self,
        call: ast.Call,
        spec: FunctionSpec,
    ) -> tuple[dict[str, ast.expr] | None, str | None]:
        if spec.has_vararg or spec.has_kwarg:
            return None, "functions with *args/**kwargs are not supported in phase-1"

        pos_values: list[ast.expr] = []
        kw_values: list[tuple[str, ast.expr]] = []

        for arg in call.args:
            if isinstance(arg, ast.Starred):
                return None, "starred positional arguments are not supported"
            pos_values.append(copy.deepcopy(arg))

        for kw in call.keywords:
            if kw.arg is None:
                return None, "**kwargs are not supported"
            kw_values.append((kw.arg, copy.deepcopy(kw.value)))

        posonly_params = [a.arg for a in spec.posonly]
        pos_or_kw_params = [a.arg for a in spec.pos_or_kw]
        kwonly_params = [a.arg for a in spec.kwonly]
        all_positional_params = posonly_params + pos_or_kw_params

        if len(pos_values) > len(all_positional_params):
            return None, "too many positional arguments"

        bound: dict[str, ast.expr] = {}
        for index, value in enumerate(pos_values):
            param = all_positional_params[index]
            bound[param] = value

        for kw_name, kw_value in kw_values:
            if kw_name in posonly_params:
                return None, "positional-only parameter passed as keyword"
            if kw_name in bound:
                return None, "duplicate argument binding"
            if kw_name in all_positional_params or kw_name in kwonly_params:
                bound[kw_name] = kw_value
            else:
                return None, f"unknown keyword argument: {kw_name}"

        n_defaults = len(spec.defaults)
        pos_defaults_map: dict[str, ast.expr] = {}
        if n_defaults > 0:
            default_param_names = all_positional_params[-n_defaults:]
            for pname, default_expr in zip(default_param_names, spec.defaults):
                pos_defaults_map[pname] = default_expr

        for param in all_positional_params:
            if param not in bound:
                if param in pos_defaults_map:
                    bound[param] = copy.deepcopy(pos_defaults_map[param])
                else:
                    return None, f"missing required positional argument: {param}"

        kw_defaults_map: dict[str, ast.expr | None] = {
            p: d for p, d in zip(kwonly_params, spec.kw_defaults)
        }
        for param in kwonly_params:
            if param not in bound:
                default_expr = kw_defaults_map.get(param)
                if default_expr is None:
                    return None, f"missing required keyword-only argument: {param}"
                bound[param] = copy.deepcopy(default_expr)

        return bound, None


def load_target_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


_INLINER_CACHE: dict[tuple[str, str], tuple[DeterministicInliner, list[str]]] = {}

_INLINE_INTERFACE_STATS: dict[str, Any] = {
    "inline_code_calls": 0,
    "functions": {},
}


def reset_inline_interface_stats() -> None:
    _INLINE_INTERFACE_STATS["inline_code_calls"] = 0
    _INLINE_INTERFACE_STATS["functions"] = {}


def _ensure_function_stats(target_name: str) -> dict[str, Any]:
    functions = _INLINE_INTERFACE_STATS["functions"]
    existing = functions.get(target_name)
    if existing is not None:
        return existing

    created = {
        "total_calls": 0,
        "supported_context_calls": 0,
        "transformed_calls": 0,
        "simply_transformer_calls": 0,
        "no_prefix_inlines": 0,
        "prefixed_inlines": 0,
        "skipped_calls": 0,
        "skipped_unsupported_context_calls": 0,
        "skipped_shadowed_calls": 0,
        "skipped_signature_calls": 0,
        "skipped_reasons": {},
        "prefixed_reasons": {},
        "signature_error_reasons": {},
        "context_counts": {},
    }
    functions[target_name] = created
    return created


def _merge_reason_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def _accumulate_interface_stats(stats: InlineStats) -> None:
    target_stats = _ensure_function_stats(stats.target)
    target_stats["total_calls"] += stats.total_calls
    target_stats["supported_context_calls"] += stats.supported_context_calls
    target_stats["transformed_calls"] += stats.transformed_calls
    target_stats["simply_transformer_calls"] += stats.simply_transformer_calls
    target_stats["no_prefix_inlines"] += stats.no_prefix_inlines
    target_stats["prefixed_inlines"] += stats.prefixed_inlines
    target_stats["skipped_calls"] += stats.skipped_calls
    target_stats["skipped_unsupported_context_calls"] += stats.skipped_unsupported_context_calls
    target_stats["skipped_shadowed_calls"] += stats.skipped_shadowed_calls
    target_stats["skipped_signature_calls"] += stats.skipped_signature_calls
    _merge_reason_counts(target_stats["skipped_reasons"], stats.skipped_reasons)
    _merge_reason_counts(target_stats["prefixed_reasons"], stats.prefixed_reasons)
    _merge_reason_counts(target_stats["signature_error_reasons"], stats.signature_error_reasons)
    _merge_reason_counts(target_stats["context_counts"], stats.context_counts)


def get_inline_interface_stats() -> dict[str, Any]:
    functions = _INLINE_INTERFACE_STATS["functions"]
    function_items = sorted(functions.items(), key=lambda item: item[0])
    return {
        "inline_code_calls": _INLINE_INTERFACE_STATS["inline_code_calls"],
        "functions": {name: data for name, data in function_items},
    }


def write_inline_interface_stats(output_path: Path) -> Path:
    resolved_path = output_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = get_inline_interface_stats()
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved_path


def _resolve_common_file(common_lib: Path) -> Path:
    candidate = common_lib
    if candidate.is_dir():
        candidate = candidate / "common.py"
    candidate = candidate.resolve()
    assert candidate.exists(), f"common library file not found: {candidate}"
    return candidate


def _resolve_to_inline_file() -> Path:
    to_inline_path = (Path(__file__).resolve().parent / "to_inline.txt").resolve()
    assert to_inline_path.exists(), f"to_inline file not found: {to_inline_path}"
    return to_inline_path

ONLY_COMMON: tuple[DeterministicInliner, list[str]] | None = None

def _get_cached_inliner(common_lib: Path) -> tuple[DeterministicInliner, list[str]]:
    global ONLY_COMMON
    if ONLY_COMMON is not None:
        return ONLY_COMMON
    common_file = _resolve_common_file(common_lib)
    to_inline_file = _resolve_to_inline_file()
    cache_key = (str(common_file), str(to_inline_file))
    cached = _INLINER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    target_names = load_target_names(to_inline_file)
    inliner = DeterministicInliner(common_file=common_file, target_names=target_names)
    cached = (inliner, target_names)
    _INLINER_CACHE[cache_key] = cached
    ONLY_COMMON = cached
    return cached


def inline_code(code: str, common_lib: Path) -> str:
    """Inline target helper calls in code using deterministic inlining."""
    inliner, target_names = _get_cached_inliner(common_lib)
    _INLINE_INTERFACE_STATS["inline_code_calls"] += 1
    transformed = code
    for target_name in target_names:
        transformed, stats = inliner.inline_target(transformed, target_name)
        _accumulate_interface_stats(stats)
    return transformed


def _normalize_grid_like(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _extract_io_pair(example: Any) -> tuple[Any, Any]:
    if hasattr(example, "x") and hasattr(example, "y"):
        return _normalize_grid_like(example.x), _normalize_grid_like(example.y)
    if isinstance(example, (list, tuple)) and len(example) == 2:
        return _normalize_grid_like(example[0]), _normalize_grid_like(example[1])
    assert False, "Unsupported IO pair format"


def check_inlined_code_io_pairs(
    inlined_code: str,
    common_lib: Path,
    examples: list[Any],
    timeout: int = 10,
) -> bool:
    """Check whether inlined code still matches expected outputs for given examples."""
    from check_inline_equivalence import run_worker

    common_file = _resolve_common_file(common_lib)
    common_dir = common_file.parent

    executable_code = inlined_code
    if "def main(" not in executable_code and "def transform(" in executable_code:
        executable_code = executable_code.replace("def transform(", "def main(", 1)

    if "def main(" not in executable_code:
        return False

    with tempfile.TemporaryDirectory(prefix="inline_check_") as temp_dir:
        temp_file = Path(temp_dir) / "candidate.py"
        temp_file.write_text(executable_code, encoding="utf-8")

        for example in examples:
            input_grid, expected_output = _extract_io_pair(example)
            result = run_worker(
                temp_file,
                common_dir,
                0,
                "main",
                input_grid=input_grid,
                timeout=timeout,
            )
            if "error" in result:
                return False
            if result.get("grid") != expected_output:
                return False

    return True


def iter_problem_files(
    source_dir: Path,
    limit: int | None = None,
    selected_files: list[str] | None = None,
) -> list[Path]:
    skip_prefixes = ("common", "inline", "count_common_calls", "check_inline_equivalence")

    if selected_files is None:
        files = [
            p
            for p in sorted(source_dir.glob("*.py"))
            if not p.name.startswith(skip_prefixes)
        ]
    else:
        files = []
        seen: set[Path] = set()
        source_dir_resolved = source_dir.resolve()
        for raw_name in selected_files:
            candidate = Path(raw_name)
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".py")

            if candidate.is_absolute():
                candidate_path = candidate.resolve()
                if not candidate_path.is_relative_to(source_dir_resolved):
                    raise ValueError(f"selected file outside source-dir: {raw_name}")
            else:
                candidate_path = (source_dir / candidate).resolve()

            if not candidate_path.exists():
                raise FileNotFoundError(f"selected file does not exist: {raw_name}")
            if candidate_path.suffix != ".py":
                raise ValueError(f"selected file is not a .py file: {raw_name}")
            if candidate_path.name.startswith(skip_prefixes):
                continue

            if candidate_path not in seen:
                seen.add(candidate_path)
                files.append(candidate_path)

    if limit is not None:
        files = files[:limit]
    return files


def apply_inlining(
    inliner: DeterministicInliner,
    source_dir: Path,
    target_dir: Path,
    target_names: list[str],
    limit: int | None = None,
    selected_files: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[Path]]:
    target_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, Any]] = []
    transformed_paths: list[Path] = []
    source_files = iter_problem_files(source_dir, limit=limit, selected_files=selected_files)

    for source_file in tqdm(source_files, total=len(source_files), desc="Inlining files"):
        code = source_file.read_text(encoding="utf-8")
        transformed_code = code
        per_target_stats: list[dict[str, Any]] = []

        for target_name in target_names:
            transformed_code, stats = inliner.inline_target(transformed_code, target_name)
            per_target_stats.append(stats.as_dict())
            if verbose:
                case_labels: list[str] = []
                if stats.simply_transformer_calls > 0:
                    case_labels.append("no-prefix")
                if stats.prefixed_inlines > 0:
                    case_labels.append("prefixed")
                if stats.skipped_calls > 0:
                    case_labels.append("error")
                if not case_labels:
                    case_labels.append("no-op")
                print(
                    "[inline] "
                    f"file={source_file.name} "
                    f"func={target_name} "
                    f"cases={','.join(case_labels)} "
                    f"transformed={stats.transformed_calls} "
                    f"simple={stats.simply_transformer_calls} "
                    f"prefixed={stats.prefixed_inlines} "
                    f"errors={stats.skipped_calls}"
                )

        out_file = target_dir / source_file.name
        out_file.write_text(transformed_code, encoding="utf-8")
        transformed_paths.append(out_file)

        report.append(
            {
                "file": source_file.name,
                "target_stats": per_target_stats,
            }
        )

    return report, transformed_paths


def compile_check(paths: list[Path]) -> list[dict[str, Any]]:
    for path in paths:
        code = path.read_text(encoding="utf-8")
        compile(code, str(path), "exec")
    return []


def validate_examples(
    source_dir: Path,
    target_dir: Path,
    common_dir: Path,
    jobs: int,
    limit: int | None = None,
    selected_files: list[str] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    from check_inline_equivalence import _load_examples, run_worker

    problem_files = iter_problem_files(source_dir, limit=limit, selected_files=selected_files)
    problem_names = [p.name for p in problem_files]

    testable = []
    untestable = []
    timeout_int = max(1, int(round(timeout)))
    for name in problem_names:
        if not (source_dir / name).with_suffix(".json").exists():
            untestable.append(name)
            continue
        if not (target_dir / name).exists():
            untestable.append(name)
            continue
        testable.append(name)

    def _shape(grid: Any) -> Any:
        if grid is None:
            return None
        if not grid or not isinstance(grid, list):
            return "?"
        first_row = grid[0]
        if not isinstance(first_row, list):
            return "?"
        return (len(grid), len(first_row))

    def compare_with_baseline(name: str) -> dict[str, Any]:
        source_file = source_dir / name
        inline_file = target_dir / name
        examples_file = (source_dir / name).with_suffix(".json")

        examples = _load_examples(examples_file)

        baseline_issues: list[str] = []
        inline_issues: list[str] = []
        for pair_idx, (input_grid, expected_output_grid) in enumerate(examples, 1):
            source_out = run_worker(
                source_file,
                common_dir,
                0,
                "main",
                input_grid=input_grid,
                timeout=timeout_int,
            )
            inline_out = run_worker(
                inline_file,
                common_dir,
                0,
                "main",
                input_grid=input_grid,
                timeout=timeout_int,
            )

            if "error" in source_out:
                baseline_issues.append(f"pair={pair_idx} source main error: {source_out['error']}")
            else:
                source_grid = source_out.get("grid")
                if source_grid != expected_output_grid:
                    baseline_issues.append(
                        f"pair={pair_idx} source main MISMATCH: "
                        f"got shape={_shape(source_grid)} expected shape={_shape(expected_output_grid)}"
                    )

            if "error" in inline_out:
                inline_issues.append(f"pair={pair_idx} inlined main error: {inline_out['error']}")
                continue

            inline_grid = inline_out.get("grid")
            if inline_grid != expected_output_grid:
                inline_issues.append(
                    f"pair={pair_idx} inlined main MISMATCH: "
                    f"got shape={_shape(inline_grid)} expected shape={_shape(expected_output_grid)}"
                )

        if baseline_issues:
            return {
                "status": "baseline-failed",
                "baseline_issues": baseline_issues,
                "inline_issues": inline_issues,
            }

        if inline_issues:
            return {"status": "regression", "inline_issues": inline_issues}

        return {"status": "ok"}

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, jobs),
        thread_name_prefix="",
        initializer=None,
        initargs=(),
    ) as executor:
        futures = {
            executor.submit(compare_with_baseline, name): name
            for name in testable
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating examples"):
            name = futures[future]
            results[name] = future.result()

    status_counter = Counter(r.get("status", "error") for r in results.values())
    regressions = {
        name: result
        for name, result in results.items()
        if result.get("status") == "regression"
    }
    baseline_failures = {
        name: result
        for name, result in results.items()
        if result.get("status") == "baseline-failed"
    }

    return {
        "testable_count": len(testable),
        "untestable_count": len(untestable),
        "untestable": sorted(untestable),
        "status_counts": dict(status_counter),
        "regressions": regressions,
        "baseline_failures": baseline_failures,
    }


def install_inline_wrappers(inliner: DeterministicInliner, target_names: list[str]) -> None:
    def make_wrapper(target_name: str):
        def wrapper(code_str: str) -> str:
            transformed, _stats = inliner.inline_target(code_str, target_name)
            return transformed

        wrapper.__name__ = f"inline_{target_name}"
        wrapper.__doc__ = f"Inline call sites for common function '{target_name}'."
        return wrapper

    for name in target_names:
        globals()[f"inline_{name}"] = make_wrapper(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-file", default="synthetic_problems-inline/common.py")
    parser.add_argument("--to-inline-file", default="to_inline.txt")
    parser.add_argument("--source-dir", default="synthetic_problems")
    parser.add_argument("--target-dir", default="synthetic_problems-inline")
    parser.add_argument("--common-dir", default="synthetic_problems-inline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--py-files",
        nargs="+",
        default=None,
        help="Specific .py files to inline under source-dir (e.g. test0.py).",
    )
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--report-file", default="inline_report.json")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    common_file = Path(args.common_file).resolve()
    to_inline_file = Path(args.to_inline_file).resolve()
    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    common_dir = Path(args.common_dir).resolve()

    target_names = load_target_names(to_inline_file)
    inliner = DeterministicInliner(common_file=common_file, target_names=target_names)
    install_inline_wrappers(inliner, target_names)

    transform_report, transformed_paths = apply_inlining(
        inliner=inliner,
        source_dir=source_dir,
        target_dir=target_dir,
        target_names=target_names,
        limit=args.limit,
        selected_files=args.py_files,
        verbose=args.verbose,
    )

    compile_errors = compile_check(transformed_paths)

    validation: dict[str, Any] | None = None
    if not args.skip_validation:
        validation = validate_examples(
            source_dir=source_dir,
            target_dir=target_dir,
            common_dir=common_dir,
            jobs=args.jobs,
            limit=args.limit,
            selected_files=args.py_files,
            timeout=args.timeout,
        )

    totals = defaultdict(int)
    aggregate_skipped_reasons: Counter[str] = Counter()
    aggregate_prefixed_reasons: Counter[str] = Counter()
    aggregate_signature_error_reasons: Counter[str] = Counter()
    aggregate_context_counts: Counter[str] = Counter()
    for file_item in transform_report:
        for s in file_item["target_stats"]:
            totals["total_calls"] += s["total_calls"]
            totals["supported_context_calls"] += s["supported_context_calls"]
            totals["transformed_calls"] += s["transformed_calls"]
            totals["simply_transformer_calls"] += s["simply_transformer_calls"]
            totals["no_prefix_inlines"] += s["no_prefix_inlines"]
            totals["prefixed_inlines"] += s["prefixed_inlines"]
            totals["skipped_calls"] += s["skipped_calls"]
            totals["skipped_unsupported_context_calls"] += s["skipped_unsupported_context_calls"]
            totals["skipped_shadowed_calls"] += s["skipped_shadowed_calls"]
            totals["skipped_signature_calls"] += s["skipped_signature_calls"]
            aggregate_skipped_reasons.update(s.get("skipped_reasons", {}))
            aggregate_prefixed_reasons.update(s.get("prefixed_reasons", {}))
            aggregate_signature_error_reasons.update(s.get("signature_error_reasons", {}))
            aggregate_context_counts.update(s.get("context_counts", {}))

    report = {
        "config": {
            "common_file": str(common_file),
            "to_inline_file": str(to_inline_file),
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "common_dir": str(common_dir),
            "limit": args.limit,
            "py_files": args.py_files,
            "jobs": args.jobs,
            "timeout": args.timeout,
            "verbose": args.verbose,
        },
        "aggregate": {
            **dict(totals),
            "skipped_reasons": dict(aggregate_skipped_reasons),
            "prefixed_reasons": dict(aggregate_prefixed_reasons),
            "signature_error_reasons": dict(aggregate_signature_error_reasons),
            "context_counts": dict(aggregate_context_counts),
        },
        "files": transform_report,
        "compile_errors": compile_errors,
        "validation": validation,
    }

    report_file = Path(args.report_file).resolve()
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report to {report_file}")
    print(
        "Aggregate calls: "
        f"total={totals['total_calls']} "
        f"transformed={totals['transformed_calls']} "
        f"simple={totals['simply_transformer_calls']} "
        f"skipped={totals['skipped_calls']}"
    )
    print(
        "Inline readability split: "
        f"no-prefix={totals['no_prefix_inlines']} prefixed={totals['prefixed_inlines']}"
    )
    print(
        "Prefixed reasons: "
        f"{dict(aggregate_prefixed_reasons)}"
    )
    print(
        "Skipped reasons: "
        f"{dict(aggregate_skipped_reasons)}"
    )
    print(
        "Signature skip details: "
        f"{dict(aggregate_signature_error_reasons)}"
    )
    print(f"Compile errors: {len(compile_errors)}")
    if validation is not None:
        print(f"Validation status counts: {validation['status_counts']}")
        print(f"Untestable files: {validation['untestable_count']}")
        print(f"Baseline failures: {len(validation['baseline_failures'])}")
        print(f"Inline regressions: {len(validation['regressions'])}")

    ok = len(compile_errors) == 0
    if validation is not None:
        ok = ok and len(validation["regressions"]) == 0

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
