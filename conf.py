args = None

Beat2Change_Opt = {
    'name': 'beat2_change',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211212_f3_meta_re/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '3Beats_1', '3Beats_2', '3Beats_3'],
    'num_classes': 4,
    'raw': True,
    'scale': False,
}

Beat3Change_Opt = {
    'name': 'beat3_change',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211212_f3_meta_re/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '3Beats_1', '3Beats_2', '3Beats_3'],
    'num_classes': 4,
    'raw': True,
    'scale': False,
}

Beat4Change_Opt = {
    'name': 'beat4_change',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 0.5,
    'file_path': './dataset/20211213_f4_meta/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '4Beats_1', '4Beats_2', '4Beats_3', '4Beats_4'],
    'num_classes': 5,
    'raw': True,
    'scale': False,
}

Beat3Change_Demo_Opt = {
    'name': 'beat_change_demo',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 1.0,
    'file_path': './dataset/20211212_f3_meta_re/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '3Beats_1', '3Beats_2', '3Beats_3'],
    'num_classes': 4,
    'raw': True,
    'scale': False,
}

Beat4Change_Demo_Opt = {
    'name': 'beat4_change_demo',
    'label_index': 6,
    'batch_size': 32,
    'sampling_rate': 399,
    'seq_len': 133,
    'overlap_ratio': 1.0,
    'file_path': './dataset/20211213_f4_meta/accgyro.csv',
    'input_dim': 6,
    'learning_rate': 0.1,
    'weight_decay': 0.00001,
    'momentum': 0.9,
    'classes': ['Not Change', '4Beats_1', '4Beats_2', '4Beats_3', '4Beats_4'],
    'num_classes': 5,
    'raw': True,
    'scale': False,
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

Beat3Change_LSTM_Opt = {
    'activation': 'dropout',
    'batch_size': 32,
    'hidden_dim': 100,
    'linear_arch': 'light',
    'loss': 'l1',
    'lr': 0.001,
    'num_lstm_layers': 1,
    'tanh': False,
    'classes': ['Not Change', '3Beats_1', '3Beats_2', '3Beats_3'],
    'num_classes': 4,
    'bidirectional': False,
}

Beat4Change_LSTM_Opt = {
    'activation': 'dropout',
    'batch_size': 32,
    'hidden_dim': 100,
    'linear_arch': 'light',
    'loss': 'l1',
    'lr': 0.001,
    'num_lstm_layers': 3,
    'tanh': False,
    'classes': ['Not Change', '4Beats_1', '4Beats_2', '4Beats_3', '4beats_4'],
    'num_classes': 5,
    'bidirectional': False,
}

IMU_Process = {
    'sync': 0, # Sync index
    'acc_sampling_rate': 399,
    'gyro_sampling_rate': 399,
}