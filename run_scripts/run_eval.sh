# python eval_batch.py


pred_dir='exp/abeta_30/igrt_debug_sphere/2023_05_15_03_50_49/reconstruction_plots/seq_0_obj'
gt_dir='datasets/abeta_30_gt'
out='exp/abeta_30/igrt_debug_sphere/2023_05_15_03_50_49/metrics_result_abeta.txt'
s=53.55457153320313
t=-26.175035

python eval_batch.py \
    -p ${pred_dir} \
    -g ${gt_dir} \
    -o ${out} \
    -s ${s} \
    -t ${t}