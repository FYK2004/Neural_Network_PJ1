import math


class scheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        raise NotImplementedError


class CosineAnnealingLR(scheduler):


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
