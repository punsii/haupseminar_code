"""Training code for this project's Gaussian Splatting models.

Builds on the gsplat library, which does the actual drawing of Gaussians to pixels and
the bookkeeping for adding and removing them. A graphics card is required: gsplat
compiles its code for the GPU the first time it runs and has no version that works on a
plain processor. Use hmgpu's default nix shell, not `.#cpu`.

Two trainers:

    python -m realtime_training.train    one random photo per step, the ordinary recipe
    python -m realtime_training.stream   photos one at a time in order, as if arriving live
"""
