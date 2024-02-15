from data_processor import *
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from model.loader import load_sentences, update_tag_scheme, parse_config
from model.layered_model import Model, Evaluator, Updater


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

pathdata = "dataset"
viSRLprocessor = Vi_SRL_processor(pathdata, vi_tokenizer)

tokenizer = viSRLprocessor.tokenizer
label2id = viSRLprocessor.label2id
id2label = viSRLprocessor.label2id
train_df, dev_df, test_df = viSRLprocessor.instances

training_set = dataset(train_df, tokenizer, MAX_LEN, label2id)
dev_set = dataset(dev_df, tokenizer, MAX_LEN, label2id)
testing_set = dataset(test_df, tokenizer, MAX_LEN, label2id)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                "early_stopping_eps" : 0,
                "early_stopping" : 5,
                # Model Settings
                "epoch" : 20,
                "replace_digit" : True,
                "lowercase" : False,
                "use_singletons" : True,


                # Network Structure
                "word_embedding_dim" : 200,
                "char_embedding_dim" : 25,
                "tag_embedding_dim" : 5,
                "batch_size" : 10,

                # Hyperparameters
                "dropout_ratio" : 0.5,
                "lr_param" : 0.0001,
                "threshold" : 5,
                "decay_rate" : 0.0001,

                # For Training and Tuning
                "gpus" : {'main': -1},
                "mode" : 'train',
                "mappings_path" : "mappings.pkl",
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }



training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

def main():
    # Init args


    model = Model(len(tokenizer.get_vocab()), len(label2id), train_params)



if __name__ == '__main__':
    main()

# model = BertForTokenClassification.from_pretrained('vinai/phobert-base-v2',
#                                                    num_labels=len(id2label),
#                                                    id2label=id2label,
#                                                    label2id=label2id)
# model.to(device)
#
# ids = training_set[0]["ids"].unsqueeze(0)
# mask = training_set[0]["mask"].unsqueeze(0)
# targets = training_set[0]["targets"].unsqueeze(0)
# ids = ids.to(device)
# mask = mask.to(device)
# targets = targets.to(device)
# outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
# initial_loss = outputs[0]
# tr_logits = outputs[1]
#
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
#
# # Defining the training function on the 80% of the dataset for tuning the bert model
# def train(epoch):
#     tr_loss, tr_accuracy = 0, 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     tr_preds, tr_labels = [], []
#     # put model in training mode
#     model.train()
#
#     for idx, batch in enumerate(training_loader):
#
#         ids = batch['ids'].to(device, dtype = torch.long)
#         mask = batch['mask'].to(device, dtype = torch.long)
#         targets = batch['targets'].to(device, dtype = torch.long)
#
#         outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
#         loss, tr_logits = outputs.loss, outputs.logits
#         tr_loss += loss.item()
#
#         nb_tr_steps += 1
#         nb_tr_examples += targets.size(0)
#
#         if idx % 100==0:
#             loss_step = tr_loss/nb_tr_steps
#             print(f"Training loss per 100 training steps: {loss_step}")
#
#         # compute training accuracy
#         flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
#         active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
#         flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
#         # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
#         active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
#         targets = torch.masked_select(flattened_targets, active_accuracy)
#         predictions = torch.masked_select(flattened_predictions, active_accuracy)
#
#         tr_preds.extend(predictions)
#         tr_labels.extend(targets)
#
#         tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
#         tr_accuracy += tmp_tr_accuracy
#
#         # gradient clipping
#         torch.nn.utils.clip_grad_norm_(
#             parameters=model.parameters(), max_norm=MAX_GRAD_NORM
#         )
#
#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     epoch_loss = tr_loss / nb_tr_steps
#     tr_accuracy = tr_accuracy / nb_tr_steps
#     print(f"Training loss epoch: {epoch_loss}")
#     print(f"Training accuracy epoch: {tr_accuracy}")
#
#
# for epoch in range(EPOCHS):
#     print(f"Training epoch: {epoch + 1}")
#     train(epoch)
