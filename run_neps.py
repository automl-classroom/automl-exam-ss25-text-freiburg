import argparse
from pathlib import Path
import warnings
import yaml

import neps

from run import main_loop


def evaluate_pipeline(
	pipeline_directory,
	previous_pipeline_directory,
	**kwargs
) -> float:
	# crucial handling of hidden dim variable
	_hidden = kwargs.pop("hidden_dim")
	kwargs.update({
		"lstm_hidden_dim": _hidden,
		"ffnn_hidden": _hidden,
	})
	# crucial handling of vocab size category types
	kwargs.update({"vocab_size": int(kwargs["vocab_size"])})
	# crucial handling of output path
	kwargs.update({"output_path": pipeline_directory})
	# crucial handling of data path
	kwargs.update({"data_path": Path(kwargs["data_path"])})

	# main call to training loop
	return main_loop(**kwargs)


def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--neps_config_path", required=True)
	parser.add_argument("--root_directory", required=True)
	parser.add_argument("--data_path", required=True)
	parser.add_argument("--seed", type=int, default=None, help="If specified, uses this seed.")
	parser.add_argument("--overwrite", action="store_true", help="If specified, overwrites `root_directory.")
	parser.add_argument("--max_evals", type=int, default=None, help="If specified, uses this.")
	parser.add_argument("--optimizer", type=str, default=None, help="If specified, uses this.")
	
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	
	# load config for HPO
	with open(Path(args.neps_config_path), "r") as f:
		neps_config = yaml.safe_load(f)

	# update seed
	_seed = None
	if "seed" in neps_config:
		neps_config["seed"] = neps_config["seed"] if args.seed is None else args.seed
		_seed = neps_config["seed"]
	if "seed" in neps_config["pipeline_space"]:
		neps_config["pipeline_space"]["seed"] = (
			neps_config["pipeline_space"]["seed"] 
			if args.seed is None 
			else args.seed
		)
		_seed = neps_config["pipeline_space"]["seed"]
	else:
		warnings.warn(f"Not using seed {args.seed} as nowhere to apply it")
	
	# update other required args
	neps_config["optimizer"] = neps_config["optimizer"] if args.optimizer is None else args.optimizer
	neps_config["overwrite_working_directory"] = (
		neps_config["overwrite_working_directory"]
		if args.overwrite is None 
		else args.overwrite
	)
	neps_config["max_evaluations_total"] = (
		neps_config["max_evaluations_total"] 
		if args.max_evals is None 
		else args.max_evals
	)
	neps_config.update({
		"root_directory": Path(args.root_directory).absolute() / f"seed={_seed}",
	})
	neps_config["pipeline_space"].update({"data_path": str(Path(args.data_path).absolute())})
	print(neps_config)

	# running NePS
	neps.run(
		evaluate_pipeline=evaluate_pipeline,
		**neps_config
	)
	print(f"NePS Done: {neps_config['optimizer']} run saved in {neps_config['root_directory']}!")
# end of file