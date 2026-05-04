'import argparse
import json
import random
import traceback
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import math
import tracemalloc
import argparse
import pathlib
import signal
import subprocess
from types import FrameType
from typing import Optional, Union, List

DAO_MAPS = [
    "arena2", "arena", "brc000d", "brc100d", "brc101d", "brc200d", "brc201d", "brc202d", "brc203d", "brc204d",
    "brc300d", "brc501d", "brc502d", "brc503d", "brc504d", "brc505d", "brc997d", "brc999d", "combat2", "combat",
    "den000d", "den001d", "den005d", "den009d", "den011d", "den012d", "den020d", "den101d", "den200d", "den200n",
    "den201d", "den202d", "den203d", "den204d", "den206d", "den207d", "den308d", "den312d", "den400d", "den401d",
    "den403d", "den404d", "den405d", "den407d", "den408d", "den500d", "den501d", "den502d", "den504d", "den505d",
    "den510d", "den520d", "den600d", "den601d", "den602d", "den900d", "den901d", "den998d", "hrt000d", "hrt001d",
    "hrt002d", "hrt201d", "hrt201n", "isound1", "lak100c", "lak100d", "lak100n", "lak101d", "lak102d", "lak103d",
    "lak104d", "lak105d", "lak106d", "lak107d", "lak108d", "lak109d", "lak110d", "lak200d", "lak201d", "lak202d",
    "lak203d", "lak250d", "lak300d", "lak302d", "lak303d", "lak304d", "lak307d", "lak308d", "lak400d", "lak401d",
    "lak403d", "lak404d", "lak405d", "lak503d", "lak504d", "lak505d", "lak506d", "lak507d", "lak510d", "lak511d",
    "lak512d", "lak513d", "lak514d", "lak515d", "lak519d", "lak526d", "lgt101d", "lgt300d", "lgt600d", "lgt601d",
    "lgt602d", "lgt603d", "lgt604d", "lgt605d", "orz000d", "orz100d", "orz101d", "orz102d", "orz103d", "orz105d",
    "orz106d", "orz107d", "orz200d", "orz201d", "orz203d", "orz300d", "orz301d", "orz302d", "orz303d", "orz304d",
    "orz500d", "orz601d", "orz700d", "orz701d", "orz702d", "orz703d", "orz704d", "orz800d", "orz900d", "orz901d",
    "orz999d", "ost000a", "ost000t", "ost001d", "ost002d", "ost003d", "ost004d", "ost100d", "ost101d", "ost102d",
    "oth000d", "oth001d", "oth999d", "rmtst01", "rmtst03", "rmtst"
]


def sigint_handler(sig: int, frame: FrameType) -> None:
    """
    Handles SIGINT (Ctrl-C) signal, prompting the user to confirm quitting the program.

    Parameters:
        sig (int): The signal number.
        frame (FrameType): The current stack frame (unused).

    Note:
        This is not a very secure method, so recovery isn't guaranteed once pressed

    """
    signal.signal(signal.SIGINT, sigint_handler)
    print('\nCtrl-C pressed. Do you want to quit? (y/n): ', end="")
    response = input().strip().lower()
    if response in ['y', 'yes']:
        pathlib.Path(sbatch_file_name).unlink()
        exit(0)


def progress_bar(progress, total) -> None:
    """
        Displays a progress bar in the console.

        Parameters:
            progress (int): The current progress value.
            total (int): The total value indicating completion.

        """
    percent = 100 * (progress / float(total))
    percent_to_show = int(percent) // 2
    bar = '❚' * percent_to_show + '-' * (50 - percent_to_show)
    print(f'\r|{bar}| {percent:.2f}%', end='\r')
    if progress == total:
        print()


def submit_job(run_lines: Optional[Union[List[str], str]], partition: str = 'main', job_name: str = 'job',
               runtime: str = '6-23:00:00', cpus_per_task: int = 2, mem: int = 60, suppress_output: bool = False,
               output_file: str = 'slurm-%j.out', suppress_error: bool = False, error_file: bool = None,
               dependency: Optional[str] = None, conda_env: Optional[str] = None, force_gpu: bool = False,
               chdir=None) -> int:
    """
    ""
    Submits a job to a SLURM workload manager with specified parameters.

    Parameters:
        run_lines (Union[List[str], str]): Commands to run in the job
        partition (str): The partition to run the job on. Default is 'main'.
        job_name (str): The name of the job. Default is 'job'.
        runtime (str): The job's runtime limit. Default is '1-00:00:00'.
        cpus_per_task (int): Number of CPUs per task. Default is 1.
        mem (int): Memory allocation in GB. Default is -1 (no memory specified).
        suppress_output (bool): If True, suppresses standard output. Default is False.
        output_file (str): File to write standard output. Default is 'slurm-%j.out'.
        suppress_error (bool): If True, suppresses error output. Default is False.
        error_file (bool): File to write error output. If one is not specified, it will be directed to the output_file
        dependency (Optional[str]): Job dependency.
        conda_env (Optional[str]): Conda environment to activate.
        force_gpu (bool): If True, ensures GPU is set when requested. Default is False.
        chdir: Where to put current dir.

    Note:
        Do not change partition unless you have a good reason to
        You can read more about these options here: https://slurm.schedmd.com/sbatch.html

    Returns:
        int: The job ID of the submitted job.

    Raises:
        Exception: If no commands are given to run, if GPU is not set when requested, or if an unknown error occurs.
    """
    # Write all the parameters into the temporary bash file
    with open(sbatch_file_name, 'w+', newline="\n") as f:
        f.write('#!/bin/bash\n')
        f.write(f"#SBATCH --partition {partition}\n")
        f.write(f'#SBATCH --time {runtime}\n')
        f.write(f'#SBATCH --cpus-per-task={cpus_per_task}\n')
        f.write(f'#SBATCH --job-name {job_name}\n')
        f.write(f'#SBATCH --constraint cpu256\n')
        f.write(f'#SBATCH --output {"/dev/null" if suppress_output else output_file}\n')

        if suppress_error:
            f.write('#SBATCH --error /dev/null')
        elif error_file:
            f.write(f'#SBATCH --error {error_file}')

        if dependency:
            f.write(f'#SBATCH --dependency={dependency}\n')

        if mem > 0:
            f.write(f'#SBATCH --mem={mem}G\n')

        if chdir:
            f.write(f'#SBATCH --chdir={chdir}\n')

        f.write('\n\n')

        f.write("echo `date`\n")
        f.write('echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID\n')
        f.write('echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n"\n')

        if conda_env:
            f.write('module load anaconda\n')
            f.write(f'source activate {conda_env}\n')

        # Handle the commands to run
        if not run_lines:
            raise Exception("No commands were given to be run")

        if isinstance(run_lines, str):
            run_lines = [run_lines]

        if run_lines:
            for line in run_lines:
                f.write(line)
                if not line.endswith('\n'):
                    f.write('\n')

    # Submit the job
    command = ['sbatch', sbatch_file_name, '--parsable']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    # stderr contains the message about if GPU is set or not, and stdout contains the ID of the submitted job
    if stderr.startswith('sbatch: GPU Parameter Not Set'):
        if force_gpu:
            raise Exception("GPU was not set when requested")
        else:
            job_id = stdout.split()[-1].strip()
    elif stderr.startswith('sbatch: GPU Parameter Set'):
        job_id = stdout.split()[-1].strip()
    elif not stderr.strip():
        job_id = stdout.split()[-1].strip()
    else:
        raise Exception(f"Unknown error from job submission: \n {stderr}")
    return int(job_id)


def sbatch_stp(algorithm, out_dir, mem=32):
    for i in range(100):
        jobname = f'stp_{algorithm}_i{i}'
        outname = f'{out_dir}/{jobname}.out'
        submit_job(run_lines=f'./hog2/bin/release/aij -d stp -h md -a {algorithm} -i {i} -n 1',
                   job_name=jobname,
                   output_file=outname,
                   conda_env='HOG2', mem=mem,
                   runtime='1-00:00:00')
        progress_bar(i + 1, 100)


def sbatch_pancake(algorithm, out_dir, mem=32):
    for i in range(50):
        for gap in range(4, 6):
            jobname = f'pancake_g{gap}_{algorithm}_i{i}'
            outname = f'{out_dir}/{jobname}.out'
            submit_job(run_lines=f'./hog2/bin/release/aij -d pancake -h GAP-{gap} -a {algorithm} -i {i} -n 1',
                       job_name=jobname,
                       output_file=outname,
                       conda_env='HOG2', mem=mem,
                       runtime='3-00:00:00')
            progress_bar(6 * i + gap + 1, 300)


def sbatch_dao(algorithm, out_dir, mem=16):
    for i, mapname in enumerate(DAO_MAPS):
        jobname = f'dao_{mapname}_{algorithm}'
        outname = f'{out_dir}/{jobname}.out'
        submit_job(run_lines=f'./aij -d dao -h {mapname} -a {algorithm} -i 0 -n 30',
                   job_name=jobname,
                   output_file=outname,
                   conda_env='HOG2', mem=mem,
                   runtime='1-00:00:00',
                   chdir='/home/siagl/aij/hog2/bin/release')
        progress_bar(i + 1, len(DAO_MAPS))


def sbatch_toh(algorithm, out_dir, mem=16):
    for i in range(50):
        for pdb in ['10+2', '8+4', '6+6']:
            jobname = f'oth_{pdb.replace("+", "_")}_{algorithm}_i{i}'
            outname = f'{out_dir}/{jobname}.out'
            submit_job(run_lines=f'./hog2/bin/release/aij -d toh -h {pdb} -a {algorithm} -i {i} -n 1',
                       job_name=jobname,
                       output_file=outname,
                       conda_env='HOG2', mem=mem,
                       runtime='6-00:00:00')

def main():
	algo = "IDA"
	lookaheads = [1,2,3,4]
	for snake in [False, True]:
		grids_sizes = [7,8,9] if snake else [6,7,8]
		s = 's' if snake else ''
		for i in grids_sizes:
			for j in grids_sizes:
				for per in [8,12,16,20]:
					for la in lookaheads:
						if j>=i and False: submit_job(run_lines=f"python BiHS/src/main.py --date SM_Grids --graph_type grid {'--snake ' if snake else ''}--size_of_graphs {i} {j} --per_blocked {per} --lookahead {la} --algorithms {algo}", job_name=f"j{s}{i}x{j}_{per}_{la}", output_file=f"j{s}{i}x{j}_{per}per_{algo}_{la}la.log", conda_env='bihs')
	for maze_blocks in [0,1,2]:
		for la in lookaheads:
			submit_job(run_lines=f"python BiHS/src/main.py --date mazes --graph_type maze --size_of_graphs 13 13 --per_blocked {maze_blocks} --algorithm {algo} --lookahead {la}", job_name=f"jm_{maze_blocks}_{la}", output_file=f"jm_{maze_blocks}blocks_{algo}_{la}la")
	for dim in [4,5,6,7]:
		for la in lookaheads:
			submit_job(run_lines=f"python BiHS/src/main.py --date cubes --graph_type cube --size_of_graphs {dim} {dim} --snake --algorithm {algo} --lookahead {la}", job_name=f"jc_{dim}_{la}", output_file=f"jc_{dim}d_{la}la")
if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    sbatch_file_name = 'temp.sh'
    main()
    pathlib.Path(sbatch_file_name).unlink()
