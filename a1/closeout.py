import copy,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
root=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023');a1=root/'karl-a1-20261024-results/A1';run=a1/('cpu-'+sys.argv[1]);source=root/'karl-a1-20261024'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
raw=(run/'results.json').read_bytes();d=json.loads(raw)
accounting=subprocess.check_output(['sacct','-j',sys.argv[1],'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS,NodeList','-P'],text=True)
rows=[x.split('|') for x in accounting.splitlines()[1:] if x]
parent=next(x for x in rows if x[0]==sys.argv[1]);assert parent[1]=='COMPLETED' and parent[2]=='0:0',accounting
elapsed=int(parent[3]);cpus=int(parent[4]);assert 'gres/gpu' not in parent[5]
receipt={'owner':'jat-a1-karl-20261024','job_id':int(sys.argv[1]),'state':parent[1],'exit_code':parent[2],'reserved_seconds':elapsed,'allocated_cpus':cpus,'reserved_cpu_hours':elapsed*cpus/3600,'gpu_hours':0,'accounting':accounting,'first_and_terminal_marker_log':str(a1/('cpu-'+sys.argv[1]+'.slurm.log')),'all_owned_jobs_terminal':True,'source':str(source),'stage':d['provenance']['stage'],'cpu_results_sha256':hashlib.sha256(raw).hexdigest(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(run/'terminal-receipt.json',receipt);dump(a1/'terminal-receipt.json',receipt)
# Numerical reduction output remains unchanged; this node record adds field meanings and accounting.
node=copy.deepcopy(d);node['record_type']='A1_KARL_component_reconciliation';node['cpu_reduction_results']={'path':str(run/'results.json'),'sha256':hashlib.sha256(raw).hexdigest()};node['new_cpu_accounting']=receipt
node['unverified_reference_fields']={'per_image_active_slot_lists':'Not retained; source preserves original positions. No count-based proof of individual masks.','larger_batch_equality':'Not run or needed: reused native batch1 only.','paired_seed_gpu_equality':'Not measured; no evaluation RNG draw by reachable-source proof. Actual seed0 retained.','imagenet_archive_sha256':'Unavailable; exact5000 JPEG content hashes and GT crop bytes verified.'}
node['images_seen_scope']='10000 reused image-condition inference outputs; this task runs no new inference'
node['new_inference_images_seen']=0
for c in node['reconstruction'].values():
 c['recon_l1']=c['source_png_l1'];c['recon_l1_p90']=c['source_png_l1_p90'];c['recon_l1_protocol']='Source PNG[0,1] L1; clipping and uint8 truncation; raw source L1 is separate'
 c['recon_lpips']=None;c['recon_lpips_null_reason']='A1 requires source LPIPS normalizeFalse on[0,1]; shared pilot Alex[-1,1] is not measured or substituted'
 c['rfid_eval']=None;c['rfid_eval_null_reason']='Prior source paired5000 pytorch-fid is not the campaign torch-fidelity protocol; descriptive source_rfid retained separately'
node['gpu_accounting']={'new':{'h100_hours':0,'other_gpu_hours':0},'reused':{'rtx2080ti_hours':8464/3600,'job_id':2827031,'scope':'whole eleven-condition IN100 evaluation; no condition-specific allocation','h100_equivalent':None,'h100_equivalent_null_reason':'No measured conversion'},'historical_context_not_charged_as_A1_reuse':{'job_2827023':'IN1K terminal; its metrics not used for A1 gate'},'reused_cpu_aggregation':{'job_id':2828154,'cpus':2,'reserved_seconds':261,'reserved_cpu_hours':2*261/3600}}
node['plan_sha256']='71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b'
dump(a1/'results.json',node)
subprocess.run(['python3',str(source/'a1/write_report.py'),str(run)],check=True)
shutil.copy2(run/'report.md',a1/'report.md')
for f in ['resolved.yaml','metric-protocol.json','source-audit.md','pip-freeze.txt','artifacts.json']:
 shutil.copy2(run/f,a1/f)
lock=json.loads((a1/'artifacts.json').read_text());lock['derived_artifacts']=[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(run.iterdir()) if f.is_file() and f.name!='artifacts.json'];lock['node_result']={'path':str(a1/'results.json'),'sha256':sha(a1/'results.json')};dump(a1/'artifacts.json',lock)
assert (run/'results.json').read_bytes()==raw
sections=[x for x in (a1/'report.md').read_text().splitlines() if x.startswith('## ')]
assert sections==['## 1 Objective','## 2 Setup','## 3 Baseline','## 4 Results','## 5 Interpretation','## 6 Limitations','## 7 Next action'],sections
print(json.dumps({'output':str(a1),'cpu':receipt,'gates':node['gates'],'reconstruction':{k:{j:v for j,v in c.items() if 'histogram' not in j} for k,c in node['reconstruction'].items()}},indent=2))
