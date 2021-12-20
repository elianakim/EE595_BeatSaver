#
# CNN - Time Signature Classifier
#

python main.py --type beat_type --dataset beat_original --model beat_type_model --method Src --log_suffix ts_reproduce --feat_eng --lr 0.001 --gpu_idx 0
python main.py --type beat_type --dataset beat_original --model beat_type_model --method Src --log_suffix ts_reproduce_ds --feat_eng --lr 0.001 --gpu_idx 0 --downsample # to add downsample option

#
# CNN - Beat Detector
#

# multi-class
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts2_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts3_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts4_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4
# binary
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts2_b_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2 --binary
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts3_b_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3 --binary
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts4_b_lr0.001 --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4 --binary
# multi-class + downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts2_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2 --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts3_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3 --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts4_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4 --downsample
# binary + downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts2_b_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2 --binary --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts3_b_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3 --binary --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model --method Src --log_suffix bd_reproduce_ts4_b_lr0.001_ds --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4 --binary --downsample

#
# CNN light model - Beat Detector
#

# binary
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts2_b_lr0.001_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2 --binary
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts3_b_lr0.001_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3 --binary
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts4_b_lr0.001_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4 --binary
# binary + downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts2_b_lr0.001_ds_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 2 --binary --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts3_b_lr0.001_ds_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 3 --binary --downsample
python main.py --type beat_change --dataset beat_original --model beat_change_model_light --method Src --log_suffix bd_reproduce_ts4_b_lr0.001_ds_light --feat_eng --lr 0.001 --gpu_idx 0 --beat_type 4 --binary --downsample

#
# LSTM - Beat Detector
#
python main.py --type beat_change --dataset beat_original --model beat_change_model_lstm --method Src --log_suffix bd_reproduce_ts2_lr0.001_lstm --lr 0.001 --feat_eng --gpu_idx 0 --beat_type 2
python main.py --type beat_change --dataset beat_original --model beat_change_model_lstm --method Src --log_suffix bd_reproduce_ts3_lr0.001_lstm --lr 0.001 --feat_eng --gpu_idx 0 --beat_type 3
python main.py --type beat_change --dataset beat_original --model beat_change_model_lstm --method Src --log_suffix bd_reproduce_ts4_lr0.001_lstm --lr 0.001 --feat_eng --gpu_idx 0 --beat_type 4
