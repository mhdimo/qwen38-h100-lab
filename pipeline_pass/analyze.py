"""Extract measured counters and timeline intervals, preserving profiler units."""
import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

METRICS=['gpu__time_duration.sum','dram__bytes.sum.per_second',
         'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
         'sm__throughput.avg.pct_of_peak_sustained_elapsed',
         'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',
         'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
         'sm__warps_active.avg.pct_of_peak_sustained_active','launch__registers_per_thread',
         'launch__shared_mem_per_block_dynamic','derived__local_spilling_requests',
         'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
         'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio']

def counters(path):
    rows=list(csv.DictReader(path.open()));units=rows[0]
    result=[]
    for row in rows[1:]:
        label=re.search(r'(gdn|silu)/[^:]+',list(row.values())[4])
        if not label:raise ValueError('unlabelled kernel counter row')
        result.append({'label':label.group(),'kernel':row['Kernel Name'],
                       'metrics':{k:{'value':float(row[k]),'unit':units[k]} for k in METRICS}})
    if len(result)!=25:raise ValueError(f'expected 25 profiled kernels, got {len(result)}')
    return result

def union_duration(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative interval')
        total+=max(0,b-max(a,end if end is not None else a))
        end=max(end if end is not None else b,b)
    return total

def timeline(path):
    with sqlite3.connect(path) as c:
        ks=c.execute('select start,end,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
        if not ks:raise ValueError('no GPU kernels')
        lo,hi=ks[0][0],max(k[1] for k in ks)
        kernel_union=union_duration([(a,b) for a,b,_ in ks])
        active=union_duration([(a,b) for a,b,node in ks if node])
        totals=c.execute('select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value order by sum(k.end-k.start) desc').fetchall()
        api=c.execute('select s.value,count(*),sum(r.end-r.start),avg(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId group by s.value order by sum(r.end-r.start) desc').fetchall()
        result={'path':str(path),'kernel_count':len(ks),'kernel_span_ms':(hi-lo)/1e6,
                'kernel_union_ms':kernel_union/1e6,'no_kernel_within_span_pct':100*(1-kernel_union/(hi-lo)),
                'graph_node_count':sum(bool(k[2]) for k in ks),'graph_kernel_union_ms':active/1e6,
                'kernels':[{'name':n,'count':count,'summed_ms':t/1e6} for n,count,t in totals],
                'cuda_api':[{'name':n,'count':count,'summed_ms':t/1e6,'mean_us':avg/1e3} for n,count,t,avg in api],
                'nvtx_fixture_ranges':[]}
        # The decode workload includes an initial prefill. Select only graph
        # replays containing the recurrence, as in the earlier Torch analysis.
        pids=c.execute('select distinct globalPid from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
        if len(pids)!=1:raise ValueError('multiple GPU processes require process-qualified correlation joins')
        correlations=[r[0] for r in c.execute("select distinct k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.graphNodeId>0 and s.value like '%fused_recurrent_gated_delta_rule_packed_decode_kernel%'")]
        graph_rows=[];graph_totals={}
        for correlation in correlations:
            nodes=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.correlationId=? and k.graphNodeId>0 order by k.start',(correlation,)).fetchall()
            launch=c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.correlationId=? and s.value like 'cudaGraphLaunch%'",(correlation,)).fetchone()[0]
            if launch!=1:raise ValueError('decode graph has no unique matching CUDA graph launch')
            a,b=nodes[0][0],max(k[1] for k in nodes)
            graph_rows.append({'correlation':correlation,'kernels':len(nodes),'span_us':(b-a)/1000,
                               'kernel_union_us':union_duration([(x,y) for x,y,_ in nodes])/1000})
            for x,y,name in nodes:
                entry=graph_totals.setdefault(name,{'count':0,'summed_ms':0})
                entry['count']+=1;entry['summed_ms']+=(y-x)/1e6
        result['decode_graphs']=graph_rows
        total=sum(v['summed_ms'] for v in graph_totals.values())
        result['decode_graph_kernels']=[dict(name=name,**v,summed_gpu_time_pct=100*v['summed_ms']/total)
            for name,v in sorted(graph_totals.items(),key=lambda item:item[1]['summed_ms'],reverse=True)]
        for start,end,label in c.execute("select start,end,text from NVTX_EVENTS where text like 'gdn/%' or text like 'silu/%'"):
            selected=[(a,b) for a,b,_ in ks if a>=start and b<=end]
            if len(selected)!=20:raise ValueError(f'expected 20 fixture calls in {label}, got {len(selected)}')
            gaps=[max(0,selected[i][0]-selected[i-1][1]) for i in range(1,len(selected))]
            result['nvtx_fixture_ranges'].append({'label':label,'calls':len(selected),
                'mean_kernel_us':statistics.mean(b-a for a,b in selected)/1000,
                'mean_interkernel_gap_us':statistics.mean(gaps)/1000})
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--compute',type=Path,required=True)
    p.add_argument('--systems',type=Path,nargs='+',required=True)
    p.add_argument('--bench',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();benches=[]
    for path in args.bench:
        d=json.loads(path.read_text())
        if d['status']!='complete':raise ValueError('incomplete benchmark')
        benches.append({'path':str(path),'gpu_before':d['gpu_before'],'gpu_after':d['gpu_after'],'results':d['results']})
    data={'counters':counters(args.compute),'timelines':[timeline(path) for path in args.systems],
          'benchmarks':benches,'note':'Profiler counters/timelines are diagnostic only. No-kernel intervals can include copies, scheduling, or other activity; they are not attributed to CPU launch overhead.'}
    args.output.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
