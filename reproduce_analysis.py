"""Recompute the paper's numerical results and comparison figures without training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/"source"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=BASE/"reproduced")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR",str(args.output/".mplconfig"))
    import numpy as np
    import upright_pendulum
    from analysis.compare_reward_results import load_events,counts_from_events,arm_summary,figures
    from analysis.final_study_analysis import write_csv
    from upright_pendulum.paired_reward_results import paired_endpoint
    assert Path(upright_pendulum.__file__).resolve().is_relative_to(BASE/"source")
    for name,expected in json.loads((BASE/"data/input_checksums.json").read_text()).items():
        assert hashlib.sha256((BASE/"data"/name).read_bytes()).hexdigest()==expected,name
    seeds=list(range(20000,20100));g=np.arange(0,1001,10);t=np.arange(201)/10
    b,xb=load_events(BASE/"data/bounded_events.csv",seeds,g)
    q,xq=load_events(BASE/"data/quadratic_events.csv",seeds,g)
    bc,qc=counts_from_events(b,t),counts_from_events(q,t)
    primary=paired_endpoint(np.isfinite(b[:,-1]),np.isfinite(q[:,-1]))
    result={"bounded":arm_summary(b,xb,bc,g),"quadratic":arm_summary(q,xq,qc,g),
            "primary_paired_endpoint":primary,"posthoc_descriptive_transitions":{},
            "capped_time_difference_s":float((np.minimum(b[:,-1],20)-np.minimum(q[:,-1],20)).mean())}
    for name,e in (("bounded",b),("quadratic",q)):
        a=np.isfinite(e[:,60]);z=np.isfinite(e[:,-1]);c=np.isfinite(e).sum(axis=0)
        result["posthoc_descriptive_transitions"][name]={"first_measured_generation_at_least_50":next(int(k) for k,n in zip(g,c) if n>=50),
            "g600_to_g1000":{"success_both":int((a&z).sum()),"success_to_failure":int((a&~z).sum()),
                              "failure_to_success":int((~a&z).sum()),"failure_both":int((~a&~z).sum())}}
    expected=json.loads((BASE/"data/reported_summary.json").read_text())
    for name in ("bounded","quadratic","primary_paired_endpoint"):
        assert result[name]==expected[name],name
    assert abs(result["capped_time_difference_s"]-expected["descriptive_capped_time_difference_s"])<1e-12
    exploratory=json.loads((BASE/"data/exploratory_descriptive_analysis.json").read_text())
    for name,value in result["posthoc_descriptive_transitions"].items():assert value==exploratory[name]
    np.savez_compressed(args.output/"reproduced_surface.npz",generations=g,evaluation_times_s=t,
                        bounded_counts=bc,quadratic_counts=qc,difference_percentage_points=bc-qc)
    write_csv(args.output/"endpoint_comparison.csv",[{"generation":int(k),"bounded_successes":int(bc[-1,j]),
              "quadratic_successes":int(qc[-1,j]),"difference_pp":int(bc[-1,j]-qc[-1,j])} for j,k in enumerate(g)])
    figures(args.output,g,t,bc,qc,b,q,primary)
    result["validation"]="All reported arm statistics, paired endpoint and exploratory transitions reproduced exactly"
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","paired_seeds":100,"evaluations_per_arm":10100,
                      "bounded":int(bc[-1,-1]),"quadratic":int(qc[-1,-1]),
                      "difference_pp":primary["difference_percentage_points"],"output":str(args.output)}))


if __name__=="__main__":main()
