"""Execute every notebook in a fresh kernel using this exact Python interpreter."""
from pathlib import Path
import sys,json,tempfile,time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
def main():
    status=[]
    with tempfile.TemporaryDirectory(prefix='ml-kernel-') as tmp:
        kernel=Path(tmp)/'portfolio';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Portfolio validation','language':'python'}),encoding='utf8')
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            start=time.perf_counter();print('Executing',path.name,flush=True)
            nb=nbformat.read(path,as_version=4)
            for cell in nb.cells:
                if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
            manager=KernelManager(kernel_name='portfolio',kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            client=NotebookClient(nb,km=manager,timeout=300,allow_errors=False,resources={'metadata':{'path':str(ROOT)}})
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            for cell in nb.cells:
                if cell.cell_type=='code':
                    assert cell.execution_count is not None
                    assert all(o.output_type!='error' for o in cell.outputs)
                cell.metadata.pop('execution',None)
            nbformat.validate(nb)
            nbformat.write(nb,path)
            status.append({'notebook':path.name,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'status':'passed','seconds':round(time.perf_counter()-start,2)})
    (ROOT/'reports/notebook_execution.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print('PASS: all',len(status),'notebooks executed end to end',flush=True)
if __name__=='__main__':main()
