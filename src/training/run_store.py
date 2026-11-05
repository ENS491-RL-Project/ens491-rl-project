"""Durable, isolated records for every v2 training attempt (not exact resume)."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import uuid
import warnings
import zipfile

import torch

from src.types import NodeStatus

REPO = Path(__file__).resolve().parents[2]
_active_run: ContextVar = ContextVar('active_training_run', default=None)


def policy_hash(policy) -> str | None:
    state_dict = getattr(policy, 'state_dict', None)
    if not callable(state_dict):
        return None
    h = hashlib.sha256()
    for name, tensor in sorted(state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if hasattr(value, 'as_dict'):
        return _json_value(value.as_dict())
    if hasattr(value, 'spec'):
        spec = value.spec
        return _json_value(spec() if callable(spec) else spec)
    return f'{type(value).__module__}.{type(value).__qualname__}'


def write_json(path: Path, value) -> None:
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    partial.replace(path)


class TrainingRun:
    def __init__(self, node, config, root=None):
        root = Path(root or os.environ.get('ENS491_RUNS_ROOT', REPO / 'runs/training'))
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex
        self.path = root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.node, self.model = node, None
        seen, children = set(), []
        def visit(child):
            if id(child) in seen:
                return
            seen.add(id(child))
            children.append({'id': child.id, 'spec_hash': child.spec_hash(),
                             'policy_sha256': policy_hash(child.policy)})
            for descendant in (*child.children, *child.lateral_sources):
                visit(descendant)
        for child in (*node.children, *node.lateral_sources):
            visit(child)
        hashes = {}
        with zipfile.ZipFile(self.path / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            files = sorted([*REPO.joinpath('src').rglob('*.py'),
                            *REPO.joinpath('scripts').rglob('*.py'), REPO / 'requirements.txt'])
            for file in files:
                relative = file.relative_to(REPO).as_posix()
                data = file.read_bytes()
                hashes[relative] = hashlib.sha256(data).hexdigest()
                archive.writestr(relative, data)
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        self.manifest = {'schema_version': 1, 'run_id': self.run_id, 'state': 'running',
                         'node_spec': node.spec(), 'config': _json_value(config),
                         'code_commit': commit, 'source_sha256': hashes,
                         'children': children, 'exact_resume_supported': False}
        write_json(self.path / 'run.json', self.manifest)

    def finish(self, reason, report=None, error=None):
        if self.model is not None:
            partial = self.path / 'model.partial.zip'
            self.model.save(partial)
            partial.replace(self.path / 'model.zip')
        if callable(getattr(self.node.policy, 'state_dict', None)):
            partial = self.path / 'policy_state_dict.pt.partial'
            torch.save(self.node.policy.state_dict(), partial)
            partial.replace(self.path / 'policy_state_dict.pt')
        self.manifest.update(state=reason, node_status=self.node.status.value,
                             policy_sha256=policy_hash(self.node.policy))
        if report is not None:
            report.update(run_id=self.run_id, run_dir=str(self.path))
            write_json(self.path / 'report.json', report)
        if error is not None:
            self.manifest['error'] = {'type': type(error).__name__, 'message': str(error)}
            self.manifest['collected_timesteps'] = getattr(self.model, 'num_timesteps', None)
        write_json(self.path / 'run.json', self.manifest)
        # Written last: distinguishes a complete archive of a failed run from a partial save.
        for artifact in self.path.iterdir():
            if artifact.is_file():
                with artifact.open('rb') as f:
                    os.fsync(f.fileno())
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in self.path.iterdir() if p.is_file() and not p.name.endswith('.partial')}
        write_json(self.path / 'complete.json', {'state': reason, 'sha256': artifacts})


def track_model(model):
    """Attach immediately after construction so exceptions can archive the last model."""
    run = _active_run.get()
    if run is not None:
        run.model = model


def record_training(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        node = bound.arguments['node']
        if node.status is NodeStatus.FROZEN:
            raise RuntimeError(f'{node.id}: cannot train a FROZEN node')
        config = {k: v for k, v in bound.arguments.items() if k not in ('node', 'run_root')}
        run = TrainingRun(node, config, bound.arguments.get('run_root'))
        token = _active_run.set(run)
        try:
            report, model = function(*args, **kwargs)
            run.model = model
            run.finish('stable' if report['stable'] else 'budget_exhausted', report=report)
            return report, model
        except BaseException as exc:
            try:
                run.finish('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=exc)
            except Exception as save_error:
                warnings.warn(f'Could not archive failed run {run.path}: {save_error}')
            raise
        finally:
            _active_run.reset(token)
            if run.model is not None and run.model.get_env() is not None:
                run.model.get_env().close()
    return wrapped
