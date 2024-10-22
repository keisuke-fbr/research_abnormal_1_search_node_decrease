from tensorflow.keras.callbacks import Callback

class ConsecutiveEpochsEarlyStopping(Callback):
    def __init__(self, monitor='loss', min_delta=0, patience=0, verbose=0, restore_best_weights=True, mode="auto"):
        super(ConsecutiveEpochsEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        self.prev_value = None
        self.total_steps = None
        self.pbar = None
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        if self.params.get('steps'):
            self.total_steps = self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            self.total_steps = (self.params['samples'] // self.params['batch_size']) + int(self.params['samples'] % self.params['batch_size'] > 0)
        else:
            self.total_steps = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print(f'Epoch {epoch + 1}/{self.epochs}')
            from tqdm.auto import tqdm  # ここでインポートすることで、verbose=0 の場合はインポートしない
            if self.total_steps:
                self.pbar = tqdm(total=self.total_steps, bar_format='{n_fmt}/{total_fmt} ━━━━━━━━━━━━━━━━━━━━ {elapsed}')
            else:
                self.pbar = tqdm(bar_format='{n_fmt} ━━━━━━━━━━━━━━━━━━━━ {elapsed}')
        else:
            self.pbar = None  # verbose=0 の場合はプログレスバーを初期化しない

    def on_batch_end(self, batch, logs=None):
        if self.verbose > 0 and self.pbar:
            self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0 and self.pbar:
            self.pbar.close()

        current = logs.get(self.monitor)
        if current is None:
            return

        if self.verbose > 0:
            print(f'{self.monitor}: {current:.4f}')

        # 改善の判定と早期停止の処理
        if self.prev_value is not None:
            change = abs(self.prev_value - current)
            if change < self.min_delta:
                self.wait += 1
                if self.verbose > 0:
                    print(f'No significant change in {self.monitor} (Δ={change:.5f}). Wait count: {self.wait}/{self.patience}')
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.verbose > 0:
                        print(f'Early stopping triggered at epoch {epoch + 1}.')
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
            else:
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
        else:
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        self.prev_value = current

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Training stopped at epoch {self.stopped_epoch + 1}.')
