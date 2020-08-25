import torch

torch.cuda.empty_cache()

from pathlib import Path
import torch

from box import Box
import pandas as pd
import sys

import datetime

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_thresh, fbeta, roc_auc

import discord

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

DATA_PATH = Path('data/')
LABEL_PATH = Path('labels/')

MODEL_PATH = Path('models/')
LOG_PATH = Path('logs/')
MODEL_PATH.mkdir(exist_ok=True)

model_state_dict = None
BERT_PRETRAINED_PATH = Path('bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
FINETUNED_PATH = Path('models/output/model_out/pytorch_model.bin')
LOG_PATH.mkdir(exist_ok=True)
OUTPUT_PATH = MODEL_PATH / 'output'
OUTPUT_PATH.mkdir(exist_ok=True)

args = Box({
    "run_text": "multilabel toxic comments with freezable layers",
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "toxic_classification_lib",
    "no_cuda": False,
    "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 512,
    "do_train": False,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 6,
    "eval_batch_size": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 6,
    "warmup_proportion": 0.0,
    "local_rank": -1,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": True,
    "fp16_opt_level": "O1",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'intent',
    "model_name": 'xlnet-base-cased',
    "model_type": 'xlnet'
})

import logging

logfile = str(LOG_PATH / 'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger()

logger.info(args)

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

databunch = BertDataBunch(args['data_dir'], LABEL_PATH, args.model_name, train_file='train.csv', val_file='train.csv',
                          test_data='test.csv',
                          text_col="comment_text", label_col=label_cols,
                          batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'],
                          multi_gpu=args.multi_gpu, multi_label=True, model_type=args.model_type)

num_labels = len(databunch.labels)

metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})

learner = BertLearner.from_pretrained_model(databunch, args.model_name, metrics=metrics,
                                            device=device, logger=logger, output_dir=args.output_dir,
                                            finetuned_wgts_path=FINETUNED_PATH, warmup_steps=args.warmup_steps,
                                            multi_gpu=args.multi_gpu, is_fp16=args.fp16,
                                            multi_label=True, logging_steps=0)


class AGC(discord.Client):

    async def on_ready(self):
        self.messages = []
        self.userlist = []
        self.toxic_users = []
        self.preds = []
        print('We have logged in as {0.user}'.format(client))

    async def on_message(self, message):
        channel_id = 747157587623018526
        if message.author == client.user:
            return
        self.messages.append(str(message.content))
        self.userlist.append({str(message.content): message.author.name})
        if len(self.messages) % 10 == 0:
            prediction = learner.predict_batch(self.messages)
            for i, (m, p) in enumerate(zip(self.messages, prediction)):
                d = {0: "toxic", 1: "insult", 2: "obscene", 3: "severe_toxic", 4: "identity_hate", 5: "threat"}
                for number in range(6):
                    if p[number][1] > 0.7:
                        print("Message: '{}' sent by '{}' flagged our detection systems as '{}' with {} confidence".format(m, self.userlist[i][m], d[number], p[number][1] * 100))
                        channel = self.get_channel(channel_id)
                        await channel.send("Message: '{}' sent by '{}' flagged our detection systems as '{}' with {} confidence".format(m, self.userlist[i][m], d[number], p[number][1] * 100))

            self.messages = []
            self.userlist = []
            self.toxic_users = []

        if message.content.startswith('$hello'):
            await message.channel.send('Hello!')


client = AGC()

client.run('')
