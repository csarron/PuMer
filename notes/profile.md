

METER-VQAv2:

`python cli/bench.py profile_flops --model meter > data/flops/meter-vqa.txt`

`merge_text=0.2 merge_r=0.2 prune_r=0.2 reduce_layers=0,2,4,6  python cli/bench.py profile_flops --model meter > data/flops/meter-vqa-0246-t0.2-r0.2-p0.2.txt`

METER-VE:

`python cli/bench.py profile_flops --model meter  --task  ve  > data/flops/meter-ve.txt`

`merge_text=0.2 merge_r=0.5 prune_r=0.3 reduce_layers=0,2,4,6  python cli/bench.py profile_flops --model meter --task  ve > data/flops/meter-ve-0246-t0.2-r0.5-p0.3.txt`

METER-NLVR2:

`python cli/bench.py profile_flops  --model meter --task nlvr2 > data/flops/meter-nlvr2.txt`

`merge_text=0.2 merge_r=0.5 prune_r=0.3 reduce_layers=2,4,6  python cli/bench.py profile_flops --model meter --task nlvr2 > data/flops/meter-nlvr2-246-t0.2-r0.5-p0.3.txt`


ViLT-VQAv2:

`python cli/bench.py profile_flops  --model vilt --task vqa2 > data/flops/vilt-vqa.txt`

`merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,5,8 python cli/bench.py profile_flops  > data/flops/vilt-vqa-258-t0.2-r0.3-p0.1.txt`

ViLT-VE:

`python cli/bench.py profile_flops  --model vilt --task  ve > data/flops/vilt-ve.txt`

`merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,4,6,8 python cli/bench.py profile_flops  --model vilt --task  ve > data/flops/vilt-ve-2468-t0.2-r0.3-p0.1.txt`


ViLT-NLVR2:

`python cli/bench.py profile_flops  --model vilt --task nlvr2 > data/flops/vilt-nlvr2.txt`

`merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,5,8 python cli/bench.py profile_flops --task nlvr2 > data/flops/vilt-nlvr2-258-t0.2-r0.3-p0.1.txt`
