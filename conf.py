args = None

BeatChange_Opt = {
    'name': 'beat_change',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 119,
    'seq_len': 30,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211210_debug2/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'classes': ['Not Change', 'Change'],
    'num_classes': 2,
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
    'acc_sampling_rate': 119,
    'gyro_sampling_rate': 119
}