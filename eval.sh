resultroot="Path to generated results ..."
dataroot="Path to LAKERED_DATASET ..."
GPU_ID="1"
echo ${resultroot}
echo ${GPU_ID}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
destimages=${dataroot}'/COD10K_CAM'
echo ${destimages}
python split_validation_subset.py --src_root ${resultroot}

echo "COD"
datapath=${resultroot}'/image_subset/COD'
fidelity --gpu 0 --kid --fid --input1 ${datapath}  --input2 ${destimages}

echo "SOD"
datapath=${resultroot}'/image_subset/SOD'
fidelity --gpu 0 --kid --fid --input1 ${datapath}  --input2 ${destimages}

echo "SEG"
datapath=${resultroot}'/image_subset/SEG'
fidelity --gpu 0 --kid --fid --input1 ${datapath}  --input2 ${destimages}

echo "Overall"
datapath=${resultroot}'/images'
fidelity --gpu 0 --kid --fid --input1 ${datapath}  --input2 ${destimages}