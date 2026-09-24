"""Load two pinned upstream architecture files into an isolated module namespace.

The upstream repository does not publish a root LICENSE at the pinned revision,
so we do not redistribute its source. Downloads are hash-checked before import.
Only optional FlashAttention training imports are replaced; SDPA is unchanged.
"""
import hashlib
import os
from pathlib import Path
import sys
import types
from urllib.request import urlopen

REVISION = 'a6db0f6135e18f4a5dad6f38df3947ed61960742'
HASHES = {
    'layers': 'b2a81c50eb3910f991403adb0cefcffea4fd79436cf75700b5f6116339127c08',
    'dmft': '0c8a0685f54b7b6febbe981e9cc3c327a33c4c03add140543877bab497fd8b60',
}


def load_architecture():
    namespace = '_synthart_dmf_' + REVISION
    if namespace + '.dmft' in sys.modules:
        return sys.modules[namespace + '.dmft'].DMFT_models
    cache = Path(os.environ.get('SYNTHART_CACHE', Path.home() / '.cache' / 'synthart')) / REVISION
    cache.mkdir(parents=True, exist_ok=True)
    package = types.ModuleType(namespace)
    package.__path__ = [str(cache)]
    sys.modules[namespace] = package
    for name, expected in HASHES.items():
        path = cache / f'{name}.py'
        if not path.exists():
            url = f'https://raw.githubusercontent.com/kyungmnlee/dmf/{REVISION}/models/{name}.py'
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Upstream DMF source hash mismatch')
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'DMF source cache hash mismatch: {path}')
        source = data.decode()
        if name == 'layers':
            source = source.replace('from .flash_attention_2_jvp import flash_attn_func as fa2_func',
                'def fa2_func(*args, **kwargs):\n    raise RuntimeError("Use torch_sdpa for inference")')
            source = source.replace('from .flash_attention_3_jvp import flash_attn_func as fa3_func', 'fa3_func = fa2_func')
        module = types.ModuleType(namespace + '.' + name)
        module.__package__ = namespace
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return module.DMFT_models
