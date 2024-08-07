import argparse
import importlib
import time
import numpy as np
from cProfile import run
import pstats
from pstats import SortKey
from tqdm import tqdm

global actions
global env

def test_performance(env, actions, duration):
    tick = 0
    num_actions = actions.shape[0]
    start = time.time()
    with tqdm(total=duration, desc="Running performance test") as pbar:
        while time.time() - start < duration:
            atns = actions[tick % num_actions]
            env.step(atns)
            tick += 1
            if tick % 100 == 0:
                pbar.update(time.time() - start - pbar.n)

    print(f'SPS: {atns.shape[0] * tick / (time.time() - start):.2f}')

def main():
    parser = argparse.ArgumentParser(description="Performance testing script")
    parser.add_argument('--profile', action='store_true', help='Run with cProfile')
    parser.add_argument('--env', type=str, default='examples.forage.Forage', help='Environment class to use')
    parser.add_argument('--num_agents', type=int, default=20, help='Number of agents')
    parser.add_argument('--duration', type=int, default=20, help='Duration of test')

    args = parser.parse_args()

    module_name, class_name = args.env.rsplit('.', 1)
    module = importlib.import_module(f'puffergrid.{module_name}')
    env_class = getattr(module, class_name)

    global env
    env = env_class(num_agents=args.num_agents)
    env.reset()

    global actions
    actions = np.random.randint(0, env.action_space.nvec, (1024, env.num_agents(), 2), dtype=np.uint32)

    if args.profile:
        print("""
            You might need to recompile with profiling:
            1. edit setup.py:compiler_directives:profile=True \n
            2. rm -rf build \n
            3. python setup.py build_ext --inplace
            """)

        cmd = f"test_performance(env, actions, {args.duration})"
        print("Running with cProfile: ", cmd)
        run(cmd, 'stats.profile')
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(25)
    else:
        test_performance(env, actions, args.duration)
        print(env.stats())

if __name__ == "__main__":
    main()
