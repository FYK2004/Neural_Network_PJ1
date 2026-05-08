import numpy as np
import os
from tqdm import tqdm


def _iter_layers(model):
    layers = getattr(model, 'layers', None)
    if layers is None:
        return []
    return layers


def _set_training(model, training: bool):
    for layer in _iter_layers(model):
        if hasattr(layer, 'training'):
            layer.training = training


def _forward_eval(model, X):
    _set_training(model, False)
    out = model(X)
    _set_training(model, True)
    return out


def _scheduler_step_batch(scheduler):
    if scheduler is None:
        return
    step_on = getattr(scheduler, 'step_on', 'iter')
    if step_on == 'iter':
        scheduler.step()


def _scheduler_step_epoch(scheduler):
    if scheduler is None:
        return
    step_on = getattr(scheduler, 'step_on', 'iter')
    if step_on != 'epoch':
        return
    if hasattr(scheduler, 'step_epoch'):
        scheduler.step_epoch()


def _scheduler_bind_training(scheduler, *, num_epochs: int, train_size: int, batch_size: int):
    if scheduler is None:
        return
    if hasattr(scheduler, 'bind_training'):
        iters_per_epoch = int(train_size / batch_size) + 1
        scheduler.bind_training(num_epochs=int(num_epochs), iters_per_epoch=int(iters_per_epoch))

class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, eval_interval=0, eval_per_iter=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        # eval_interval: 0=每 epoch 末 dev；>0=每 N iter；eval_per_iter 等价于 1。
        self.eval_interval = 1 if eval_per_iter else int(eval_interval)

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        best_score = 0

        X0, y0 = train_set
        assert X0.shape[0] == y0.shape[0]
        _scheduler_bind_training(
            self.scheduler,
            num_epochs=int(num_epochs),
            train_size=int(X0.shape[0]),
            batch_size=int(self.batch_size),
        )

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            num_iterations = int(X.shape[0] / self.batch_size) + 1
            dev_score, dev_loss = None, None
            last_dev_score, last_dev_loss = None, None
            for iteration in range(num_iterations):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                eval_logits = _forward_eval(self.model, train_X)
                trn_score = self.metric(eval_logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()

                self.optimizer.step()
                _scheduler_step_batch(self.scheduler)

                if self.eval_interval > 0:
                    if (iteration % self.eval_interval) == 0 or last_dev_score is None:
                        dev_score, dev_loss = self.evaluate(dev_set)
                        last_dev_score, last_dev_loss = dev_score, dev_loss
                    self.dev_scores.append(last_dev_score)
                    self.dev_loss.append(last_dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}", flush=True)
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}", flush=True)
                    if self.eval_interval > 0:
                        print(f"[Dev] loss: {last_dev_loss}, score: {last_dev_score}", flush=True)

            if self.eval_interval == 0:
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.extend([dev_score] * num_iterations)
                self.dev_loss.extend([dev_loss] * num_iterations)
            else:
                if dev_score is None:
                    dev_score, dev_loss = self.evaluate(dev_set)

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(
                    f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}",
                    flush=True,
                )
                best_score = dev_score

            _scheduler_step_epoch(self.scheduler)
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = _forward_eval(self.model, X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)