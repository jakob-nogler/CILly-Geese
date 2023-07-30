import argparse
import json
from training_process import start_training
import os

parser = argparse.ArgumentParser(
    prog="Trains the best CIL collaborative filtering model"
)

parser.add_argument('--config', default="oneidea.json",
                    help="Configuration file used")
parser.add_argument('--verbose', default=False, action="store_true",
                    help="Print detailed information on the training process")
parser.add_argument("--full_data", default=False,
                    action="store_true", help="Use the full data set for training")
parser.add_argument("--log", default=False, action="store_true",
                    help="Save the training process logs to config")
parser.add_argument("--save_models", default=False,
                    action="store_true", help="Save the trained models")
parser.add_argument("--resume", default=False, action="store_true",
                    help="Restarts the learning process from the previously saved checkpoint")
parser.add_argument("--wandb", default=False, action="store_true",
                    help="Use wandb sweeps hyperparameter search")
parser.add_argument("--sweep_id", default=None,
                    help="Restart an existing sweep")
parser.add_argument("--save_test_predictions", action="store_true",
                    help="Save the validation set predictions to file")
parser.add_argument("--almost_full_data", action="store_true",
                    help="Use 0.995 of the set for training")
parser.add_argument("--movielens", action="store_true",
                    help="Use the movielens dataset")


args = parser.parse_args()

if args.resume:
    with open(f"./checkpoints/checkpoint.json") as f:
        config = json.load(f)
else:
    with open(f"./hyperparameters/{args.config}", 'r') as f:
        config = json.load(f)

os.makedirs("./results", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./trained_models", exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./test_results", exist_ok=True)

start_training(config,
               verbose=args.verbose,
               use_full_data=args.full_data,
               save_log=args.log,
               save_models=args.save_models,
               use_wandb=args.wandb,
               sweep_id=args.sweep_id,
               save_validation=args.save_test_predictions,
               almost_full_data=args.almost_full_data,
               movielens=args.movielens)
try:
    os.remove("./checkpoints/checkpoint.json")
except:
    pass
