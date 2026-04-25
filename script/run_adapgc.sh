
corruption_list=( "audio"  "video" )
for corruption in "${corruption_list[@]}"; do
    python run_adapgc.py \
        --dataset 'ks50' \
        --json-root 'json_csv_files/ks50/' \
        --label-csv 'json_csv_files/class_labels_indices_ks50.csv' \
        --pretrain_path 'checkpoints/cav_mae_ks50.pth' \
        --tta-method 'ADAPGC' \
        --severity-start 5 \
        --severity-end 5 \
        --corruption-modality ${corruption} \
        --batch-size 16 \
        --exp-name ADAPGC \
        --gpu '0,1' \
        --w-c 0.01 \
        --w-g 1 \
        --gamma 1 
done
