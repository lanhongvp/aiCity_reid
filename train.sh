CUDA_VISIBLE_DEVICES=0 python train_aicity.py --height 256 --width 256 -a densePCB --save-dir log_debug --lr 0.0002 --use_pcb --resume log_nls_slr/best_model.pth.tar 
