import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import diffeq_layers

from models.spatial.cnf import *


class TVCNF(nn.Module):
    def __init__(self, odefunc, dim, reg=0.0):
        super().__init__()
        self.s_start = 0
        self.s_end = 1
        self.func = odefunc
        self.reg_rate = reg
        self.dim = 2

    def integrate(
        self, t0, t1, x, logpx, tol=None, method=None, norm=None, intermediate_states=0
    ):
        """
        Args:
            t0: (N,)
            t1: (N,)
            x: (N, ...)
            logpx: (N,)
        """
        e = torch.randn_like(x[:, : self.dim])
        l2_norm = torch.zeros(2)
        state = (t0, t1, x, logpx, e, l2_norm)
        tt = [0, 1]
        tt = torch.tensor(tt).to(x)

        solution = odeint(self, state, tt, method="dopri5")

        _, _, z, logpz, _, l2 = solution
        z, logpz = z[-1], logpz[-1]

        l2_reg = self.reg_rate * l2.sum()
        return z, logpz + l2_reg

    def forward(self, s, state):
        t0, t1, x, _, e, _ = state

        with torch.enable_grad():
            x = x.requires_grad_(True)

            ratio = (t1 - t0) / (1 - 0)
            t = s * ratio + t0

            dx_t = self.func(t, x)

            dx_s = dx_t * ratio.unsqueeze(1)

            vjp = torch.autograd.grad(dx_s, x, e, create_graph=True, retain_graph=True)[
                0
            ]

            dlogpx = (vjp * e).sum(1)

            l2_norm = (
                torch.tensor(((dlogpx * dlogpx).sum(), (dx_s * dx_s).sum()))
                / t0.shape[0]
            )

        result = (
            torch.zeros_like(t0),
            torch.zeros_like(t1),
            dx_s,
            -dlogpx,
            torch.zeros_like(e),
            l2_norm,
        )
        return result


class JumpODEFunc(nn.Module):
    def __init__(self, func, dim, aux_dim, aux_odefunc, time_offset):
        super().__init__()
        self.func = func
        self.dim = dim
        self.aux_dim = aux_dim
        self.aux_odefunc = aux_odefunc
        self.time_offset = time_offset

    def forward(self, t, state):
        x, h = state[:, : self.dim], state[:, self.dim :]
        aux = h[:, : self.aux_dim]
        x = torch.cat((x, aux), dim=1)
        dx = self.aux_odefunc(t, dx)
        dh = torch.zeros_like(h)
        return torch.cat((dx, dh), axis=1)


class TemporalODEFunc(nn.Module):
    def __init__(self, hdim, dstate_fn, intensity_fn):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn
        self.intensity_fn = intensity_fn

    def forward(self, t, state):
        lam, h = state
        dh = self.dstate_fn(t, h)
        int = self.get_intensity(h)

        return int, dh

    def get_intensity(self, tpp_state):
        return torch.sigmoid(self.intensity_fn(tpp_state[:, : self.hdim]))


def comp_CNFs():

    func = build_fc_odefunc(
        dim=2,
        hidden_dims=[64, 64, 64],
        layer_type="concat",
        actfn="swish",
        zero_init=False,
    )

    while True:
        torch.manual_seed(0)
        A = TimeVariableCNF(func, 2)
        torch.manual_seed(0)
        # B = TimeVariableCNF(func, 2)
        B = TVCNF(func, 2)

        data = torch.rand((128 * 48, 2))
        t0 = torch.rand((128 * 48))
        t1 = torch.zeros((128 * 48))
        logpx = torch.zeros_like(t0)

        Ay, Alogpx = A.integrate(t0, t1, data, logpx)
        By, Blogpx = B.integrate(t0, t1, data, logpx)

        mean_diff_y = torch.mean(Ay - By)
        mean_diff_log = torch.mean(Alogpx - Blogpx)

        print(
            "difference in y:",
            mean_diff_y.item(),
            "logpx :",
            mean_diff_log.item(),
        )


if __name__ == "__main__":
    comp_CNFs()
