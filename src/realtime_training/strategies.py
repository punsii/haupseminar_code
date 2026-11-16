"""Deciding where to add and remove Gaussians, with one interface over gsplat's two ways.

A scene starts as a sparse cloud of points from COLMAP - far too few, and in the wrong
places, to look good. Training therefore has to keep editing the population: splitting
Gaussians where the image is still blurry, deleting ones that have faded to invisible.
gsplat calls this a "strategy", and ships two. Pick with --strategy:

  default  The approach from the original Gaussian Splatting paper. Where the rendering
           is still wrong, it splits or copies Gaussians to add detail; Gaussians that
           have become nearly transparent get deleted. Two consequences: the population
           grows without any ceiling, and it periodically dims every Gaussian at once as
           a way of forcing the useless ones to reveal themselves. It stops editing the
           population partway through training and just polishes what is left.
           Reproduces published quality figures, so use it for the offline baseline.

  mcmc     A later method that treats training as random exploration rather than greedy
           splitting (MCMC = Markov-chain Monte Carlo, the family of algorithms it
           borrows from). It holds the Gaussian count at a fixed ceiling: instead of
           adding new ones it moves faded ones to where they are needed, and jitters
           positions slightly to help them settle. Because the count never grows, memory
           use and the time to draw a frame both stay flat - which is what a real-time
           system needs. Needs two extra penalty terms in the loss to stay stable.

The two have slightly different calling conventions, which this wrapper hides:
  - only `default` wants the scene size when setting up
  - only `mcmc` wants the current position learning rate on each step
  - only `mcmc` needs the extra penalty terms (see regularizer())
"""
from gsplat.strategy import DefaultStrategy, MCMCStrategy


class Densifier:
    def __init__(self, kind, extent, total_steps, cap_max=1_000_000,
                 opacity_reg=0.01, scale_reg=0.01, absgrad=False, verbose=False):
        self.kind = kind
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg

        if kind == "default":
            # Scale the reference schedule (30k-step run) to whatever budget we have.
            s = max(total_steps / 30_000, 1e-3)
            self.strategy = DefaultStrategy(
                refine_start_iter=int(500 * s),
                refine_stop_iter=int(15_000 * s),
                refine_every=max(1, int(100 * s)),
                reset_every=max(1, int(3_000 * s)),
                absgrad=absgrad,
                verbose=verbose,
            )
            self.state = self.strategy.initialize_state(scene_scale=extent)
        elif kind == "mcmc":
            s = max(total_steps / 30_000, 1e-3)
            self.strategy = MCMCStrategy(
                cap_max=cap_max,
                refine_start_iter=int(500 * s),
                refine_stop_iter=int(25_000 * s),
                refine_every=max(1, int(100 * s)),
                verbose=verbose,
            )
            self.state = self.strategy.initialize_state()
        else:
            raise ValueError(f"unknown strategy {kind!r} (expected 'default' or 'mcmc')")

    def check(self, splats):
        self.strategy.check_sanity(splats.params, splats.opts)

    def regularizer(self, splats):
        """Extra penalty terms added to the loss.

        Nothing for `default`. The mcmc strategy needs a mild push towards low opacity and
        small size, otherwise it drifts towards a few huge, fully opaque blobs.
        """
        if self.kind != "mcmc":
            return 0.0
        p = splats.params
        return (self.opacity_reg * p["opacities"].sigmoid().abs().mean()
                + self.scale_reg * p["scales"].exp().abs().mean())

    def pre_backward(self, splats, step, info):
        """Call after rendering but before loss.backward().

        This asks gsplat to hold on to a quantity it would otherwise discard: how much
        each Gaussian wants to move across the screen. That is the signal used later to
        decide where the image still lacks detail.
        """
        self.strategy.step_pre_backward(splats.params, splats.opts, self.state, step, info)

    def post_backward(self, splats, step, info):
        """Call after loss.backward() and the optimizer step. This is where the Gaussian
        population actually changes size."""
        if self.kind == "mcmc":
            self.strategy.step_post_backward(
                splats.params, splats.opts, self.state, step, info, lr=splats.means_lr(),
            )
        else:
            self.strategy.step_post_backward(
                splats.params, splats.opts, self.state, step, info, packed=splats.packed,
            )
