### arguments

import os , argparse
import config

def train_args(args_list=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
    #                     default=os.path.join(config.data_folder,'Models',config.default_model_checkpoint),type=str)
    parser.add_argument('-m', '--model_checkpoint', action='store', dest='model_checkpoint',
                    default=config.default_model_checkpoint,type=str)
    parser.add_argument('--data_folder', action='store', dest='data_folder',
                        default=config.data_folder,type=str) 
    parser.add_argument('--train_file', action='store', dest='train_file',
                        default=os.path.join(config.data_folder,'retrieval_train_data_gpt.xlsx'),type=str) 
    parser.add_argument('--eval_file', action='store', dest='eval_file',
                        default=os.path.join(config.data_folder,'retrieval_train_data_clean.xlsx'),type=str) 
    parser.add_argument('--model_folder', action='store', dest='model_folder',
                        default=config.model_folder,type=str)
    parser.add_argument('--model_outdir', 
                        action='store', 
                        dest='MODEL_OUTDIR',
                        default=os.path.join(config.model_folder,
                                 'retrieval_{}'.format(config.default_model_checkpoint)),
                                type=str)
    parser.add_argument('--checkpoint_folder', 
                        action='store', 
                        dest='checkpoint_folder',
                        default=os.path.join(config.model_folder,
                                 'training_{}_checkpoints'.format(config.default_model_checkpoint)),
                                type=str)
    parser.add_argument('--cache_dir', action='store', dest='cache_dir',
                        default=os.path.join(config.cache_dir,'cache'),type=str) 
    parser.add_argument('--batch_size', action='store', dest='batch_size',
                        default=16,type=int) 
    # parser.add_argument('--vocab_aug', dest='vocab_aug',action='store_true')
    # parser.add_argument('--no_vocab_aug', dest='vocab_aug',action='store_false')
    parser.set_defaults(vocab_aug=True)
    if args_list is not None:
        args = parser.parse_args(args_list) 
    else:
        args = parser.parse_args()  
    return args