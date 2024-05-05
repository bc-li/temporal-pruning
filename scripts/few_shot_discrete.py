import os
import numpy as np
from subprocess import Popen, PIPE
from skopt import Optimizer
from skopt.space import Integer
from skopt.utils import use_named_args
from eval_diffuseq import evaluate
import argparse
import math
import subprocess
import re
from collections import Counter
import time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for Diffusion model"
    )
    parser.add_argument("--model_path", type=str, help="model name")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps")
    parser.add_argument("--cuda_device", type=int, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--task", type=str, help="task name")
    parser.add_argument("--run_script", type=str, help="run which script")
    args = parser.parse_args()
    return args


args = parse_arguments()

timesteps_arg = args.timesteps
cuda_device = args.cuda_device
reparam_model_file_path = args.model_path

task_name = args.task
task_name_eval = task_name.replace("few-shot-", "")
reparam_model_folder = os.path.dirname(reparam_model_file_path)
reparam_model_file_name = os.path.basename(reparam_model_file_path)
reparam_max_diffusion_steps = 49
best_bleu_so_far = 0
best_timesteps_so_far = None
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

log_file_name = (
    f"{reparam_model_file_name}_timesteps_{args.timesteps}_{current_time}.log"
)


def write_to_log_file(msg):
    print("writing to logfile at: ", log_file_name)
    with open(log_file_name, "a") as f:
        f.write(msg + "\n")


def generate_individual():
    timesteps = [49]
    step_size = (49 - 0) // (timesteps_arg - 1)
    for i in range(timesteps_arg - 1):
        timesteps.append(timesteps[-1] - step_size)
    return timesteps


def run_decode_ddim_eval_mt(timesteps):
    timesteps_str = ",".join(map(str, timesteps))
    cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_device} bash mt_generate.sh \
        -a false \
        -c {reparam_model_file_path} \
        -d {task_name_eval} \
        -h {timesteps_str} \
        -i {timesteps_arg-1} \
        -x test
    """
    process = Popen(cmd, shell=True)
    process.wait()


def run_decode_ddim_opt_mt(timesteps):
    timesteps_str = ",".join(map(str, timesteps))
    cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_device} bash mt_generate.sh \
        -a false \
        -c {reparam_model_file_path} \
        -d {task_name} \
        -h {timesteps_str} \
        -i {timesteps_arg-1} \
        -x valid
    """
    process = Popen(cmd, shell=True)
    process.wait()


def run_decode_ddim_eval_diffuseq(timesteps):
    timesteps_str = ",".join(map(str, timesteps))
    cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_device} bash diffuseq_generate.sh \
        -a false \
        -b true \
        -c {reparam_model_file_path} \
        -d {task_name_eval} \
        -h {timesteps_str} \
        -i {timesteps_arg-1} \
        -x test
    """
    process = Popen(cmd, shell=True)
    process.wait()


def run_decode_ddim_opt_diffuseq(timesteps):
    timesteps_str = ",".join(map(str, timesteps))
    cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_device} bash diffuseq_generate.sh \
        -a false \
        -b true \
        -c {reparam_model_file_path} \
        -d {task_name} \
        -h {timesteps_str} \
        -i {timesteps_arg-1} \
        -x valid
    """
    process = Popen(cmd, shell=True)
    process.wait()


def evaluate_mt():
    cmd = f"""
        cd .. && \
        cd .. && \
        bash scripts/compound_split_bleu.sh {reparam_model_folder}/generate.out
    """
    output = subprocess.check_output(cmd, shell=True, text=True)
    print(output)
    pattern = r"BLEU4 = (\d+\.\d+)"
    matches = re.findall(pattern, output)
    if matches:
        bleu4_score = float(matches[0])
        print("BLEU4 score:", bleu4_score)
        return bleu4_score
    else:
        print("No BLEU4 score found in the output")
        exit()


def fitness_function(timesteps, mode="opt"):

    global best_bleu_so_far, best_timesteps_so_far
    print(f"Running with timesteps: {timesteps}")
    try:
        if args.run_script == "mt":
            if mode == "opt":
                run_decode_ddim_opt_mt(timesteps)
                bleu_score = evaluate_mt()
            if mode == "eval":
                run_decode_ddim_eval_mt(timesteps)
                bleu_score = evaluate_mt()
        elif args.run_script == "diffuseq":
            if mode == "opt":
                run_decode_ddim_opt_diffuseq(timesteps)
                bleu_score = evaluate(folder=reparam_model_folder)
            if mode == "eval":
                run_decode_ddim_eval_diffuseq(timesteps)
                bleu_score = evaluate(folder=reparam_model_folder)
        if mode == "opt":
            if bleu_score > best_bleu_so_far:
                best_bleu_so_far = bleu_score
                best_timesteps_so_far = timesteps

    except Exception as e:
        print(f"Exception encountered: {e}")
        bleu_score = 0

    log_msg_opt = f"[OPT] Current Timesteps: {timesteps}, BLEU score: {bleu_score}, Best Timesteps: {best_timesteps_so_far}, Best BLEU score: {best_bleu_so_far}"
    log_msg_eval = f"[EVAL] Current Timesteps: {timesteps}, BLEU score: {bleu_score}, Best Timesteps: {best_timesteps_so_far}, Best BLEU score: {best_bleu_so_far}"

    if mode == "opt":
        print(log_msg_opt)
        write_to_log_file(log_msg_opt)
    else:
        print(log_msg_eval)
        write_to_log_file(log_msg_eval)

    return (bleu_score,)


def create_space():
    space = []
    space.append(Integer(48, 50, name=f"x_0"))
    max_val = reparam_max_diffusion_steps
    for i in range(timesteps_arg - 1):
        upper_bound = max_val
        lower_bound = 0
        space.append(Integer(lower_bound, upper_bound, name=f"x_{i+1}"))
        print("space", space)

    return space


def main():
    global best_bleu_so_far, best_timesteps_so_far, reparam_model_file_name

    space = create_space()

    @use_named_args(space)
    def objective(**params):
        print(params)
        timesteps = [49] + [params[f"x_{i+1}"] for i in range(timesteps_arg - 1)]
        bleu_score = fitness_function(timesteps, mode="opt")
        return -bleu_score[0]

    n_calls = 0
    write_to_log_file("Begin OPT.....\n")
    optimizer = Optimizer(dimensions=space)
    for i in range(n_calls):
        msg_iter = "[iter" + str(i) + "]"
        with open(log_file_name, "a") as f:
            f.write(msg_iter)
        if i == 0:
            timesteps = generate_individual()
            bleu_score = fitness_function(timesteps, mode="opt")
            optimizer.tell(timesteps, -bleu_score[0])
        else:
            params = optimizer.ask()
            timesteps = [params[i] for i in range(timesteps_arg)]
            counter = Counter(params)
            has_duplicate = any(value > 2 for value in counter.values())
            while has_duplicate:
                optimizer.tell(params, -1)
                params = optimizer.ask()
                print("params", params)
                print("CURRENT params contain more than 2 zeros, regenerate params")
                timesteps = [params[i] for i in range(timesteps_arg)]
                counter = Counter(params)
                has_duplicate = any(value >= 2 for value in counter.values())
            f = objective(params)
            if math.isnan(f):
                f = 0
            print(f"Current Timesteps: {params}, Current BLEU score: {-f}")
            optimizer.tell(params, f)
            print(
                f"Iteration {i + 1}/{n_calls}, Current Timesteps: {params}, Current BLEU score: {-f}, Best Timesteps: {best_timesteps_so_far}, Best BLEU score: {best_bleu_so_far}"
            )
        if i % 20 == 0 and i != 0:
            fitness_function(best_timesteps_so_far, mode="eval")

    print(f"Best BLEU score: {best_bleu_so_far}, Timesteps: {best_timesteps_so_far}")


if __name__ == "__main__":
    main()
