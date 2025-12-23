import argparse
import os, fnmatch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=int, default=0)
    parser.add_argument('--configs', '-c', type=str, default="Batch_Configs")

    parser.add_argument('--env', '-e', type=str, default="ContinuousCartpole")
    parser.add_argument('--time_scale', '-t', type=int, default=20)

    args = parser.parse_args()

    env_dir = f'../Environments/{args.env}/'
    configs_dir = f'{env_dir}Config/{args.configs}/'

    config_files = fnmatch.filter(os.listdir(configs_dir), '*.yaml')

    for i, c in enumerate(config_files):
        cmd = f"python RunPolicy.py -r {args.run} \
        -c {c} -bc {args.configs} -e {args.env} -t {args.time_scale} -ng -n _Batch_{i}"

        os.system(cmd)

if __name__ == '__main__':
    main()




