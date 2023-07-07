
for i in $(seq 1 1 5); do

for t in vqa2 ve nlvr2; do

p=data/results/bench$i-$t
mkdir -p $p


# bench meter
for b in $(seq 16 8 72); do

echo b$b-$t-meter-ori
python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-ori.txt
echo

echo b$b-$t-meter-dyvit
prune_layers=2,4,6 keep_ratio=0.5 python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-dyvit.txt
echo

echo b$b-$t-meter-246-t0.2-r0.5-p0.3
merge_text=0.2 merge_r=0.5 prune_r=0.3 reduce_layers=2,4,6 python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-246-t0.2-r0.5-p0.3.txt
echo

echo b$b-$t-meter-0246-t0.2-r0.5-p0.3
merge_text=0.2 merge_r=0.5 prune_r=0.3 reduce_layers=0,2,4,6 python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-0246-t0.2-r0.5-p0.3.txt
echo

echo b$b-$t-meter-0246-t0.2-r0.5-p0.2
merge_text=0.2 merge_r=0.5 prune_r=0.2 reduce_layers=0,2,4,6 python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-0246-t0.2-r0.5-p0.2.txt
echo

echo b$b-$t-meter-0246-t0.2-r0.5-p0.1
merge_text=0.2 merge_r=0.5 prune_r=0.1 reduce_layers=0,2,4,6 python cli/bench.py bench $b --model meter --task $t 2>&1 | tee $p/b$b-$t-meter-0246-t0.2-r0.5-p0.1.txt

done

t=irtr
# bench vilt
# for b in $(seq 32 32 640); do
for b in 32 48 64 128 192 224 256 320 480 512 640 720; do

cfg=b$b-$t-vilt-ori
echo $cfg
python cli/bench.py bench $b --model vilt --task $t 2>&1 | tee $p/$cfg.txt
echo

cfg=b$b-$t-vilt-2468-t0.2-r0.3-p0.1
echo $cfg
merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,4,6,8 python cli/bench.py bench $b --model vilt --task $t 2>&1 | tee $p/$cfg.txt
echo

cfg=b$b-$t-vilt-258-t0.2-r0.3-p0.3
echo $cfg
merge_text=0.2 merge_r=0.3 prune_r=0.3 reduce_layers=2,5,8 python cli/bench.py bench $b --model vilt --task $t  2>&1 | tee $p/$cfg.txt
echo

cfg=b$b-$t-vilt-258-t0.2-r0.3-p0.1
echo $cfg
merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,5,8 python cli/bench.py bench $b --model vilt --task $t 2>&1 | tee $p/$cfg.txt
echo

done


done

done

python cli/bench.py profile_flops > data/flops/vilt-vqa.txt
python cli/bench.py profile_flops  --model vilt --task  ve > data/flops/vilt-ve.txt

merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,5,8 python cli/bench.py profile_flops  > data/flops/vilt-vqa-258-t0.2-r0.3-p0.1.txt

merge_text=0.2 merge_r=0.3 prune_r=0.1 reduce_layers=2,4,6,8 python cli/bench.py profile_flops  --model vilt --task  ve > data/flops/vilt-ve-2468-t0.2-r0.3-p0.1.txt

python cli/bench.py profile_flops --model meter > data/flops/meter-vqa.txt

python cli/bench.py profile_flops --model meter  --task  ve  > data/flops/meter-ve.txt

merge_text=0.2 merge_r=0.2 prune_r=0.2 reduce_layers=0,2,4,6  python cli/bench.py profile_flops --model meter > data/flops/meter-vqa-0246-t0.2-r0.2-p0.2.txt

merge_text=0.2 merge_r=0.5 prune_r=0.3 reduce_layers=0,2,4,6  python cli/bench.py profile_flops --model meter --task  ve > data/flops/meter-ve-0246-t0.2-r0.5-p0.3.txt
