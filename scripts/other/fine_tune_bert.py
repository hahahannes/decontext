from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_examples = [
    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
train_loss = losses.ContrastiveLoss(model=model)

model.fit([(train_dataloader, train_loss)], show_progress_bar=True)