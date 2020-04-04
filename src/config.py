import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
BERT_MODEL = "/home/ubuntu/TAKD/bert_intent_classification/bert_model/bert_base_uncased"
MODEL_PATH = "output_model_bert_base.bin"
TRAINING_FILE = "../banking_data/train.csv"
TESTING_FILE  = "../banking_data/test.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True
)