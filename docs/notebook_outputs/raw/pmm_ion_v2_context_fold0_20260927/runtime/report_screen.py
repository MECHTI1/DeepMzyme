"""Read-only validation collection; write exploratory fold-0 reports only."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root / 'src'))
    from benchmarking.pmm_ion_campaign import CampaignPaths, PREDECLARED_CONTRASTS, forbidden_read_roots
    from benchmarking.pmm_ion_analysis import collect_units, prediction_metrics, read_prediction_rows
    from benchmarking.pmm_comparator import read_campaign_contract, validate_prediction_rows, verify_comparator_outputs
    from training.access_guard import install_forbidden_read_guard
    install_forbidden_read_guard(forbidden_read_roots(args.train_dir))
    collected = collect_units(CampaignPaths(args.campaign_dir), args.train_dir)
    arms, missing, identities = {}, [], []
    for config_id, item in collected.items():
        if 0 not in item['folds']:
            missing.append(config_id)
            continue
        unit = item['folds'][0]
        receipt, rows = unit['receipt'], unit['rows']
        run_dir = args.campaign_dir / 'runs' / unit['run_name']
        shell_support = json.loads((run_dir / 'dataset_summary.json').read_text())['first_shell_support']
        biases = {}
        if item['config'].readout != 'none':
            import torch
            checkpoint = torch.load(run_dir / 'best_model_checkpoint.pt', map_location='cpu', weights_only=False)
            biases = {key: value.detach().cpu().tolist() for key, value in checkpoint['model_state_dict'].items() if 'binding_bias' in key}
        assert receipt['campaign_run_identity']['source_tree_sha256'] == 'adc95c42261448dc9d35572a138a3b8349de79124d9708618f511c27a848dd23'
        identities.append({row['source_uid']: (row['physical_ion_id'], row['group_id'], row['y_common4']) for row in rows})
        arms[config_id] = {'run_name': unit['run_name'], 'metrics': unit['metrics'],
                          'selected_epoch': receipt['selected_epoch'],
                          'identity': receipt['campaign_run_identity'],
                          'checkpoint_sha256': receipt['selected_checkpoint_sha256'],
                          'predictions_sha256': receipt['validation_predictions']['sha256'],
                          'independent_replay': 'verified',
                          'first_shell_support': shell_support, 'learned_pooling_biases': biases,
                          'validation_parent_pockets': len({row['parent_pocket_id'] for row in rows}),
                          'validation_pdb_groups': len({row['group_id'] for row in rows})}
    assert identities and all(item == identities[0] for item in identities)
    verify_comparator_outputs(args.campaign_dir)
    manifest = json.loads((args.campaign_dir / 'pmm_comparator/pmm_comparator_manifest.json').read_text())
    pmm_path = args.campaign_dir / 'pmm_comparator/fold0_predictions.csv'
    assert hashlib.sha256(pmm_path.read_bytes()).hexdigest() == manifest['folds']['0']['predictions_sha256']
    _, membership, _ = read_campaign_contract(args.campaign_dir)
    pmm_rows = read_prediction_rows(pmm_path)
    validate_prediction_rows(pmm_rows, membership, 0)
    assert {r['source_uid']: (r['physical_ion_id'], r['group_id'], r['y_common4']) for r in pmm_rows} == identities[0]
    pmm = prediction_metrics(pmm_rows, native_labels=None)
    contrasts = []
    for label, control, challenger in PREDECLARED_CONTRASTS:
        if control.config_id not in arms or challenger.config_id not in arms:
            continue
        left, right = arms[control.config_id]['metrics']['common4'], arms[challenger.config_id]['metrics']['common4']
        contrasts.append({'contrast': label, 'balanced_accuracy_delta_pp': 100*(right['balanced_accuracy']-left['balanced_accuracy']),
                          'macro_f1_delta_pp': 100*(right['macro_f1']-left['macro_f1']),
                          'class_recall_delta_pp': {k:100*(right['recall'][k]-left['recall'][k]) for k in left['recall']}})
    report = {'schema_version':1, 'recorded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'status':'complete_single_fold_screen' if not missing else 'partial_single_fold_screen',
              'fold':0, 'model_seed':42, 'epochs':50, 'held_out_access':False,
              'completed_configurations':len(arms), 'required_configurations':9,
              'missing_configurations':missing, 'arms':arms, 'pmm_same_fold':pmm,
              'descriptive_contrasts':contrasts,
              'interpretation':'Exploratory single-fold, single-seed validation. Checkpoints selected on this fold. No promotion, confidence interval, paper-parity or superiority claim. Full grid requires all 45 fits.'}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'screen_report.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['configuration','selected_epoch','common4_balanced_accuracy','common4_macro_f1','accuracy','Mn_recall','Cu_recall','Zn_recall','Class_VIII_recall','native_balanced_accuracy','Fe_recall','Co_recall','Ni_recall']
    records=[]
    for name, arm in arms.items():
        m,n=arm['metrics']['common4'],arm['metrics']['native']
        row={'configuration':name,'selected_epoch':arm['selected_epoch'], 'common4_balanced_accuracy':m['balanced_accuracy'],'common4_macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'native_balanced_accuracy':n['balanced_accuracy']}
        row.update({label.replace(' ','_')+'_recall':value for label,value in m['recall'].items()})
        row.update({label+'_recall':n['recall'].get(label,'') for label in ('Fe','Co','Ni')})
        records.append(row)
    with (args.out_dir/'screen_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    lines=['# PMM campaign: exploratory fold-0 screen','',f"Verified configurations: {len(arms)}/9. Matched validation ions: {len(identities[0])}.",'',report['interpretation'],'',
           '| Configuration | Epoch | Common-four BA | Macro-F1 | Mn recall | Cu recall | Zn recall | VIII recall |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        values=[row[k] for k in ('common4_balanced_accuracy','common4_macro_f1','Mn_recall','Cu_recall','Zn_recall','Class_VIII_recall')]
        lines.append(f"| {row['configuration']} | {row['selected_epoch']} | "+' | '.join(f'{100*v:.3f}%' for v in values)+' |')
    m=pmm['common4']; vals=[m['balanced_accuracy'],m['macro_f1'],*m['recall'].values()]
    lines+=['| PMM released recipe, same fold | — | '+' | '.join(f'{100*v:.3f}%' for v in vals)+' |','',
            'PMM uses its released features; DeepMzyme uses the declared ESM/geometry inputs. This is a matched known-site comparison, not paper protocol reproduction.','',
            'Six-class native metrics and Fe/Co/Ni recalls are retained in the CSV/JSON; common-four predictions sum Fe+Co+Ni probabilities before argmax.','']
    if missing: lines+=['Pending: '+', '.join(missing)+'.','']
    (args.out_dir/'screen_report.md').write_text('\n'.join(lines))
    print(json.dumps({'status':report['status'],'completed':len(arms),'missing':missing}))


if __name__ == '__main__':
    main()
