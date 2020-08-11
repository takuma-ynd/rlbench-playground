from multiprocessing import Pipe
from multiprocessing import Process
import signal
import warnings

import numpy as np
from torch.distributions.utils import lazy_property

import pfrl
import time


def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env, process_idx = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.action_space, env.observation_space))
            elif cmd == "spec":
                remote.send(env.spec)
            elif cmd == "seed":
                remote.send(env.seed(data))
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
        self.env_fns = env_fns
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        self.remotes[0].send(("get_spaces", None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.closed = False
        self._mask = np.zeros(self.num_envs)
        self._timeout_steps = 60

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
        print('MASK: {}'.format(self._mask))
        self._assert_not_closed()
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            # if self._mask[i]:
            #     print('skipping remote {} due to the mask'.format(i))
            #     continue
            remote.send(("step", action))
            print('(host) {} remote.send(("step", action))'.format(i))
        results = []
        for i in range(len(self.remotes)):
            # if self._mask[i]: continue
            counter = 0
            has_response = False
            while not has_response:
                has_response = self.remotes[i].poll()
                print('step: (remote {}) remote.poll() returned {}, counter {}'.format(i, has_response, counter))
                if counter != 0 and counter % self._timeout_steps == 0:
                    print('blacklisting remote {}'.format(i))
                    self._mask[i] = True
                    print('cleaning up...')
                    print('sending "close" to remote {}'.format(i))
                    self.remotes[i].send(("close", None))
                    print('self.ps[i].terminate()')
                    self.ps[i].terminate()
                    print('re-initializing Process...')
                    # update Pipe()
                    list_remotes = list(self.remotes)
                    list_work_remotes = list(self.work_remotes)
                    list_remotes[i], list_work_remotes[i] = Pipe()
                    self.remotes = tuple(list_remotes)
                    self.work_remotes = tuple(list_work_remotes)
                    self.ps[i] = Process(target=worker, args=(self.work_remotes[i], self.env_fns[i]))
                    self.ps[i].start()
                    time.sleep(15)
                    self.remotes[i].send(("step", actions[i]))  # a bit HACKY...
                counter += 1
                if not has_response:
                    time.sleep(0.5)
            print('waiting for the responses from remote.recv()...')
            results.append(self.remotes[i].recv())
            print('waiting for the responses from remote.recv()...DONE')

        # results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def reset(self, mask=None):
        self._assert_not_closed()
        print('MASK: {}'.format(self._mask))
        # mask = self._mask
        if mask is None:
            mask = np.zeros(self.num_envs)
        for i, (m, remote) in enumerate(zip(mask, self.remotes)):
            if not m:
                remote.send(("reset", None))

        obs = []
        timeout_steps = 10
        for i in range(len(self.remotes)):
            counter = 0
            has_response = False
            while not has_response:
                has_response = self.remotes[i].poll()
                print('reset: (remote {}) remote.poll() returned {}, counter {}'.format(i, has_response, counter))
                if counter != 0 and counter % self._timeout_steps == 0:
                    print('[reset] blacklisting remote {}'.format(i))
                    self._mask[i] = True
                    print('[reset] cleaning up...')
                    print('[reset] sending "close" to remote {}'.format(i))
                    self.remotes[i].send(("close", None))
                    print('[reset] self.ps[i].terminate()')
                    self.ps[i].terminate()
                    print('[reset] re-initializing Process...')
                    # update Pipe()
                    list_remotes = list(self.remotes)
                    list_work_remotes = list(self.work_remotes)
                    list_remotes[i], list_work_remotes[i] = Pipe()
                    self.remotes = tuple(list_remotes)
                    self.work_remotes = tuple(list_work_remotes)
                    self.ps[i] = Process(target=worker, args=(self.work_remotes[i], self.env_fns[i]))
                    self.ps[i].start()
                    time.sleep(15)
                    self.remotes[i].send(("reset", None))  # a bit HACKY...
                counter += 1
                if not has_response:
                    time.sleep(0.5)
            if mask[i]:
                print('[reset] !! using self.last_obs for remote {} due to the original mask'.format(i))
                obs.append(self.last_obs[i])
                continue
            print('[reset] waiting for the responses from remote.recv()...')
            obs.append(self.remotes[i].recv())
            print('[reset] waiting for the responses from remote.recv()...DONE')

        # print('waiting for the observation responses from remote.recv()...')
        # obs = [
        #     remote.recv() if not m else o
        #     for m, remote, o in zip(mask, self.remotes, self.last_obs)
        # ]
        # print('waiting for the observation responses from remote.recv()...DONE')
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

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
            remote.send(("seed", seed))
        results = [remote.recv() for remote in self.remotes]
        return results

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"
