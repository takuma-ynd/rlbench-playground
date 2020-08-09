from multiprocessing import Pipe
from multiprocessing import Process
import signal
import warnings

import numpy as np
from torch.distributions.utils import lazy_property

import pfrl


def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env, process_idx = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                print('(remote) {} received cmd == "step"'.format(process_idx))
                ob, reward, done, info = env.step(data)
                print('(remote) env.step(data) ran successfully...')
                remote.send((ob, reward, done, info))
                print('(remote) {} remote.send((ob, reward, done, info))'.format(process_idx))
            elif cmd == "reset":
                print('(remote) {} received cmd == "reset"'.format(process_idx))
                ob = env.reset()
                print('(remote) {} env.reset() ran successfully...'.format(process_idx))
                remote.send(ob)
                print('(remote) {} remote.send(ob)'.format(process_idx))
            elif cmd == "close":
                print('(remote) {} received cmd == "close"'.format(process_idx))
                remote.close()
                break
            elif cmd == "get_spaces":
                print('(remote) {} received cmd == "get_spaces"'.format(process_idx))
                remote.send((env.action_space, env.observation_space))
                print('(remote) {} remote.send((env.action_space, env.observation_space))'.format(process_idx))
            elif cmd == "spec":
                print('(remote) {} received cmd == "spec"'.format(process_idx))
                remote.send(env.spec)
                print('(remote) {} remote.send((env.spec))'.format(process_idx))
            elif cmd == "seed":
                print('(remote) {} received cmd == "seed"'.format(process_idx))
                remote.send(env.seed(data))
                print('(remote) {} remote.send((env.seed(data)))'.format(process_idx))
            else:
                raise NotImplementedError
    finally:
        env.close()


class MultiprocessVectorEnv(pfrl.env.VectorEnv):
    """VectorEnv where each env is run in its own subprocess.

    Args:
        env_fns (list of callable): List of callables, each of which
            returns gym.Env that is run in its own subprocess.
    """

    def __init__(self, env_fns):
        if np.__version__ == "1.16.0":
            warnings.warn(
                """
NumPy 1.16.0 can cause severe memory leak in pfrl.envs.MultiprocessVectorEnv.
We recommend using other versions of NumPy.
See https://github.com/numpy/numpy/issues/12793 for details.
"""
            )  # NOQA

        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        print('(host) running p.start()...')
        for p in self.ps:
            p.start()
        print('(host) running p.start()...DONE')
        self.last_obs = [None] * self.num_envs
        print('(host) remotes[0].send(("get_spaces", None))')
        self.remotes[0].send(("get_spaces", None))
        print('waiting for the responses from remotes[0].recv()...')
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()

    @lazy_property
    def spec(self):
        self._assert_not_closed()
        self.remotes[0].send(("spec", None))
        print('waiting for the responses from remotes[0].recv()...')
        print('remotes[0]: {}'.format(self.remotes[0]))
        spec = self.remotes[0].recv()
        print('waiting for the responses from remotes[0].recv()...DONE')
        return spec

    def step(self, actions):
        self._assert_not_closed()
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            remote.send(("step", action))
            print('(host) {} remote.send(("step", action))'.format(i))
        print('waiting for the responses from remote.recv()...')
        results = [remote.recv() for remote in self.remotes]
        print('waiting for the responses from remote.recv()...DONE')
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def reset(self, mask=None):
        self._assert_not_closed()
        if mask is None:
            mask = np.zeros(self.num_envs)
        for i, (m, remote) in enumerate(zip(mask, self.remotes)):
            if not m:
                print('(host) {} remote.send(("reset", None))'.format(i))
                remote.send(("reset", None))

        print('waiting for the observation responses from remote.recv()...')
        obs = [
            remote.recv() if not m else o
            for m, remote, o in zip(mask, self.remotes, self.last_obs)
        ]
        print('waiting for the observation responses from remote.recv()...DONE')
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        print('(host) remote.send(("close", None))')
        for remote in self.remotes:
            remote.send(("close", None))
        print('(host) running p.join()...')
        for p in self.ps:
            p.join()
        print('(host) running p.join()...DONE')

    def seed(self, seeds=None):
        self._assert_not_closed()
        if seeds is not None:
            if isinstance(seeds, int):
                seeds = [seeds] * self.num_envs
            elif isinstance(seeds, list):
                if len(seeds) != self.num_envs:
                    raise ValueError(
                        "length of seeds must be same as num_envs {}".format(
                            self.num_envs
                        )
                    )
            else:
                raise TypeError(
                    "Type of Seeds {} is not supported.".format(type(seeds))
                )
        else:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            print('(host) remote.send(("seed", None))')
            remote.send(("seed", seed))
        print('waiting for the seed responses from remote.recv()...')
        results = [remote.recv() for remote in self.remotes]
        print('waiting for the seed responses from remote.recv()...DONE')
        return results

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"
