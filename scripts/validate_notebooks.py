"""Execute every notebook in a fresh kernel; errors fail the command."""
import argparse
import json
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('notebooks', nargs='*', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/executed-notebooks'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.notebooks or sorted((root / 'notebooks').glob('*.ipynb'))
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    # A temporary kernel spec guarantees the notebook uses this interpreter.
    with tempfile.TemporaryDirectory() as temp:
        kernel = Path(temp) / 'synthart'
        kernel.mkdir()
        (kernel / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='Synth-Artiste', language='python')))
        manager = KernelSpecManager(kernel_dirs=[temp])
        for path in paths:
            started = time.time()
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            from jupyter_client import KernelManager
            km = KernelManager(kernel_name='synthart', kernel_spec_manager=manager)
            client = NotebookClient(nb, timeout=3600, km=km, resources={'metadata': {'path': str(root)}})
            try:
                client.execute()
            except Exception:
                nbformat.write(nb, args.output / path.name)
                raise
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
            nbformat.write(nb, args.output / path.name)
            report.append(dict(notebook=path.name, status='passed', seconds=round(time.time()-started, 1),
                               executed_cells=sum(c.cell_type == 'code' for c in nb.cells)))
            print(report[-1], flush=True)
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
