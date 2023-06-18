import argparse


def _add_common_args(arg_parser):
    arg_parser.add_argument('--config', type=str)

    arg_parser.add_argument("--local_rank", type=int, default=-1, help="")
    arg_parser.add_argument("--world_size", type=int, default=-1, help="")

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add_argument('--store_examples', action='store_true', default=False,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add_argument('--device_id', type=int, default=-1, help="gpu device id")
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="prompt4ner", help="Type of model")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in Prompt4NER")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")

    arg_parser.add_argument('--no_overlapping', action='store_true', default=False)
    arg_parser.add_argument('--no_partial_overlapping', action='store_true', default=False)
    arg_parser.add_argument('--no_duplicate', action='store_true', default=False)
    arg_parser.add_argument('--cls_threshold', type=float, default=0.5)
    arg_parser.add_argument('--boundary_threshold', type=float, default=0.5)
    arg_parser.add_argument('--lstm_layers', type=int, default=3)
    arg_parser.add_argument('--decoder_layers', type=int, default=3)
    arg_parser.add_argument('--pool_type', type=str, default = "max")
    arg_parser.add_argument('--prompt_number', type=int, default=60)
    arg_parser.add_argument('--prompt_type', type=str, default="soft")
    arg_parser.add_argument('--prompt_length', type=int, default=3) # 2 + n
    arg_parser.add_argument('--prompt_individual_attention', action='store_true', default=False)
    arg_parser.add_argument('--sentence_individual_attention', action='store_true', default=False)
    arg_parser.add_argument('--last_layer_for_loss', type=int, default=1)
    arg_parser.add_argument('--withimage', action='store_true', default=False)
    
    # Misc
    arg_parser.add_argument('--seed', type=int, default=-1, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")


def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--save_path_include_iteration', action='store_true', default=False)

    
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--stage_one_lr_scale', type=float, default=10)
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")


    
    arg_parser.add_argument('--match_solver', type=str, help="", default="hungarian")
    arg_parser.add_argument('--type_loss', type=str, help="", default="celoss")
    arg_parser.add_argument('--match_warmup_epoch', type=int, help="", default=0)

    
    
    arg_parser.add_argument('--nil_weight', type=float, default=-1)
    arg_parser.add_argument('--match_boundary_weight', type=float, default=10.0)
    arg_parser.add_argument('--match_class_weight', type=float, default=2.0)
    arg_parser.add_argument('--loss_boundary_weight', type=float, default=2.0)
    arg_parser.add_argument('--loss_class_weight', type=float, default=2.0)

    
    arg_parser.add_argument('--deeply_weight', type=str, help="", default="same")
    arg_parser.add_argument('--use_masked_lm', action='store_true', default=False)
    arg_parser.add_argument('--repeat_gt_entities', type=int, default=-1, help="")
    arg_parser.add_argument('--eval_every_epochs', type=int, default=1, help="")
    

    arg_parser.add_argument('--split_epoch', type=int, default=0, help="")

    _add_common_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser
