args = None

BeatChange_Opt = {
    'name': 'beat_change',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211211_meta/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '3Beats_1', '3Beats_2', '3Beats_3'],
    'num_classes': 4,
    'raw': True,
    'scale': False,
    # 'classes': ['Not Change', 'Change'],
    # 'num_classes': 2,
}

BeatType_Opt = {
    'name': 'beat_type',
    'label_index': 7,
    'batch_size': 32,
    'sampling_rate': 119,
    'seq_len': 119,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211210_debug2/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'classes': ["2beat", "3beat", "4beat"],
    'num_classes': 3,
}

IMU_Process = {
    'sync': 0, # Sync index
    'acc_sampling_rate': 399,
    'gyro_sampling_rate': 399,
}