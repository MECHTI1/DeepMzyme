"""Serial launch/status/archive operations for the existing Colab allocation."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from campaign_control import alive, read, save, sha


def paths():
    setup=read('/content/pilot_setup.json')
    campaign=Path('/content/metal_coordination_geometry_pilot_v1')
    return setup,Path(setup['output_dir']),campaign,campaign/'validation_prediction_export'


def require_idle(original,campaign,output):
    for directory in (original,campaign):
        for name in ('host_step_process.json','active_process.json'):
            if alive(read(directory/name,{})):
                raise ValueError('A training/preflight worker is active; export must run serially')
    if any(alive(read(output/name,{})) for name in ('host_export_process.json','host_export_child_process.json')):
        raise ValueError('A validation export worker is already active')


def require_geometry_complete(campaign):
    attempts=read(campaign/'campaign_attempt_ledger.json',[])
    for block in ('G1','G2','GR'):
        completed={a['run_id'] for a in attempts if a['block']==block and a['status']=='completed'}
        if len(completed)!=5:
            raise ValueError(f'Geometry block {block} is incomplete')
    latest=attempts[-1]
    receipt=read(campaign/'transfer_receipts'/f"{latest['attempt_id']}.json",{})
    if not (latest['status']=='completed' and receipt.get('drive_verified') and receipt.get('local_sha256_verified')
            and receipt.get('attempt_id')==latest['attempt_id'] and receipt.get('archive_sha256')
            and receipt.get('manifest_sha256')==sha(campaign/'campaign_manifest.json')):
        raise ValueError('Archive and verify the final geometry attempt before exporting predictions')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('launch','worker','status','archive'))
    parser.add_argument('--helper-path',type=Path,default=Path('/content/export_geometry_validation.py'))
    parser.add_argument('--helper-sha256')
    parser.add_argument('--deadline-epoch',type=float)
    args=parser.parse_args()
    setup,original,campaign,output=paths()
    if args.action in ('launch','worker'):
        if not args.helper_sha256 or sha(args.helper_path)!=args.helper_sha256:
            raise ValueError('Validation exporter helper SHA256 does not match the approved artifact')
        if args.deadline_epoch is None or not math.isfinite(args.deadline_epoch) or args.deadline_epoch<=time.time():
            raise ValueError('A future finite export deadline is required')
    if args.action=='launch':
        require_idle(original,campaign,output)
        require_geometry_complete(campaign)
        previous=read(output/'validation_prediction_export.json',{})
        if previous.get('status')=='complete':
            result=dict(status='already_completed',output_dir=str(output),exported_runs=previous.get('exported_runs'))
        else:
            output.mkdir(parents=True,exist_ok=True)
            number=len(read(output/'analysis_attempts.json',[]))+1
            log=output/f'host_export_{number:03d}.log'
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--helper-path',str(args.helper_path),
                     '--helper-sha256',args.helper_sha256,'--deadline-epoch',str(args.deadline_epoch)]
            with log.open('w') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            result=dict(status='started',pid=process.pid,started_epoch=time.time(),log=str(log),command=command,
                        helper_sha256=args.helper_sha256,deadline_epoch=args.deadline_epoch)
            save(output/'host_export_process.json',result)
    elif args.action=='worker':
        command=[sys.executable,str(args.helper_path),'--root','/content/DeepMzyme_geometry_v1',
                 '--campaign-dir',str(campaign),'--output-dir',str(output),'--budget-root',str(original),
                 '--allocation-started-epoch',str(setup['allocation_started_epoch']),
                 '--deadline-epoch',str(args.deadline_epoch)]
        process=subprocess.Popen(command)
        save(output/'host_export_child_process.json',dict(pid=process.pid,started_epoch=time.time(),command=command))
        returncode=process.wait()
        report=read(output/'validation_prediction_export.json',{})
        result=dict(returncode=returncode,ended_epoch=time.time(),
                    status='completed' if returncode==0 and report.get('status')=='complete' else 'failed',
                    helper_sha256=args.helper_sha256,command=command)
        save(output/'host_export_worker_result.json',result)
    elif args.action=='status':
        process=read(output/'host_export_process.json',{})
        report=read(output/'validation_prediction_export.json',{})
        attempts=read(output/'analysis_attempts.json',[])
        result=dict(alive=alive(process) or alive(read(output/'host_export_child_process.json',{})),worker=read(output/'host_export_worker_result.json'),
                    export_status=report.get('status'),exported_runs=report.get('exported_runs',0),
                    verified_prediction_csv_count=len(list(output.glob('*_validation_predictions.csv'))),
                    latest_analysis_attempt=attempts[-1] if attempts else None,
                    analysis_budget_usage=read(original/'coordination_geometry_analysis_budget_usage.json'))
        if process.get('log') and Path(process['log']).is_file():
            result['log_tail']=Path(process['log']).read_text()[-1800:]
    else:
        require_idle(original,campaign,output)
        if not output.is_dir():
            raise ValueError('No validation export artifacts exist')
        attempts=read(output/'analysis_attempts.json',[])
        if any(a['status']=='running' for a in attempts):
            raise ValueError('Reconcile an interrupted validation export before archival')
        exports=Path('/content/verified_exports/geometry_validation')
        exports.mkdir(parents=True,exist_ok=True)
        # Reuse the established archive verifier schema in this separate namespace.
        ident=f"attempt_{len(attempts):03d}"
        archive=exports/f'{ident}.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=3) as stream:
            stream.add(output,arcname='validation_prediction_export')
            for path in (campaign/'campaign_manifest.json',original/'coordination_geometry_analysis_budget_usage.json',
                         original/'allocation_ledger.json',args.helper_path,Path(__file__).resolve()):
                if path.is_file():
                    stream.add(path,arcname=path.name)
        parts=[]
        with archive.open('rb') as stream:
            for i in range(10000):
                chunk=stream.read(16*1024*1024)
                if not chunk:
                    break
                part=archive.with_name(archive.name+f'.part{i:03d}')
                part.write_bytes(chunk)
                parts.append(dict(path=str(part),bytes=len(chunk),sha256=sha(part)))
        report=read(output/'validation_prediction_export.json',{})
        result=dict(attempt_id=ident,analysis_kind='geometry_validation_export',archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
                    manifest_sha256=sha(campaign/'campaign_manifest.json'),helper_sha256=report.get('helper_sha256'),
                    prediction_report_sha256=sha(output/'validation_prediction_export.json') if report else None,
                    export_complete=report.get('status')=='complete',parts=parts)
        save(exports/f'{ident}_archive.json',result)
    print('EXPORT_CONTROL_RESULT='+json.dumps(result))


if __name__=='__main__':
    main()
