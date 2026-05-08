from abc import abstractmethod
import math


class scheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        raise NotImplementedError


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1, step_on='iter') -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.step_on = step_on
        self.epoch_idx = 0

    def step(self) -> None:
        if self.step_on != 'iter':
            return
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

    def step_epoch(self) -> None:
        if self.step_on != 'epoch':
            return
        self.epoch_idx += 1
        if self.epoch_idx % self.step_size == 0:
            self.optimizer.init_lr *= self.gamma


class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=None, gamma=0.1, step_on='iter') -> None:
        super().__init__(optimizer)
        if milestones is None:
            milestones = []
        self.milestones = set(milestones)
        self.gamma = gamma
        self.step_on = step_on
        self.epoch_idx = 0

    def step(self) -> None:
        if self.step_on != 'iter':
            return
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

    def step_epoch(self) -> None:
        if self.step_on != 'epoch':
            return
        self.epoch_idx += 1
        if self.epoch_idx in self.milestones:
            self.optimizer.init_lr *= self.gamma


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.95, step_on='iter') -> None:
        super().__init__(optimizer)
        self.gamma = gamma
        self.step_on = step_on

    def step(self) -> None:
        if self.step_on != 'iter':
            return
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma

    def step_epoch(self) -> None:
        if self.step_on != 'epoch':
            return
        self.optimizer.init_lr *= self.gamma


class CosineAnnealingLR(scheduler):
    """
    Cosine annealing for SGD.init_lr.

    - step_on='epoch': step once per epoch (t goes 1..T_max_epochs)
    - step_on='iter': step once per training iteration (t goes 1..T_max_iters)

    LR(t) = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * t / T_max))
    """

    def __init__(self, optimizer, T_max=None, eta_min=0.0, step_on='epoch') -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = float(eta_min)
        self.step_on = step_on

        self.base_lr = float(optimizer.init_lr)
        self._t = 0
        self._T = None

    def bind_training(self, *, num_epochs: int, iters_per_epoch: int) -> None:
        if self.T_max is None:
            if self.step_on == 'epoch':
                self._T = int(num_epochs)
            else:
                self._T = int(num_epochs) * int(iters_per_epoch)
        else:
            self._T = int(self.T_max)

        if self._T <= 0:
            raise ValueError(f'CosineAnnealingLR: invalid T_max ({self._T}).')

        self.optimizer.init_lr = self.base_lr

    def _apply(self) -> None:
        if self._T is None:
            return
        t = min(self._t, self._T)
        cos_inner = math.pi * (t / self._T)
        lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(cos_inner))
        self.optimizer.init_lr = float(lr)

    def step(self) -> None:
        if self.step_on != 'iter':
            return
        self._t += 1
        self._apply()

    def step_epoch(self) -> None:
        if self.step_on != 'epoch':
            return
        self._t += 1
        self._apply()
