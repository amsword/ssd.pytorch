#CUDA_VISIBLE_DEVICES=0 ipython3 --pdb train.py -- --max_iter 1 --expid test
#CUDA_VISIBLE_DEVICES=0 ipython3 --pdb predict.py -- \
    #-p "{'pred_file': '/home/jianfw/code/quickdetection/output/voc20_ssd_A/snapshot/model_iter_120000.caffemodel.voc20.test.predict', \
         #'trained_model': 'weights/ssd300_mAP_77.43_v2.pth', \
         #'confidence_threshold': 0.01, \
         #'cuda': True, \
         #'type': 'predict'}"

#CUDA_VISIBLE_DEVICES=0 ipython3 --pdb qd_ssd.py -- \
    #-p "{'pred_file': '/home/jianfw/code/quickdetection/output/voc20_ssd_A/snapshot/model_iter_120000.caffemodel.voc20.test.predict', \
         #'trained_model': 'weights/ssd300_mAP_77.43_v2.pth', \
         #'confidence_threshold': 0.01, \
         #'cuda': True, \
         #'type': 'ssd_pipeline'}"

CUDA_VISIBLE_DEVICES=0 ipython3 --pdb qd_ssd.py -- \
    -p "{'type': 'ssd_pipeline'}"
