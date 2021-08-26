from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    AdamW,
    get_scheduler
)

import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from datasets import load_metric
from prefixtune import PrefixTuningModel
import data

import nltk
nltk.download('punkt')


def main():
    batch_size = 2
    max_source_length = 1024
    max_target_length = 256
    padding = 'max_length'

    epoch_num = 10
    optimizer_steps = 3
    base_learning_rate = 5e-5
    weight_decay = 0.01
    num_warmup_steps = 100
    lr_scheduler_type = 'linear'
    num_beams = 3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    seq2seq_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = PrefixTuningModel(tokenizer, seq2seq_model, device)

    raw_dataset = data.get_cnn_dataset()
    column_names = raw_dataset['train'].column_names

    dataset = data.process_dataset(raw_dataset, tokenizer, max_source_length, max_target_length, padding, batch_size,
                                   column_names)

    dataset.save_to_disk('content/processed_dataset')

    train_dataset, eval_dataset, test_dataset = data.train_eval_test_split(dataset)
    train_subset = data.get_subset(train_dataset, len(eval_dataset))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=seq2seq_model,
        padding=padding
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    subset_dataloader = DataLoader(
        train_subset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    num_training_steps = epoch_num * len(train_dataloader) // optimizer_steps

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=base_learning_rate,
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    train_losses = []
    train_losses_epoch = []
    eval_losses_epoch = []

    completed_steps = 0
    model.to(device)

    for epoch in range(epoch_num):
        train_loss_sum = 0
        eval_loss_sum = 0
        loss_buf = 0

        print('model training')
        model.train()

        for step, batch in enumerate(train_dataloader):
            completed_steps += 1

            batch = batch.to(model.device)
            outputs = model.forward(**batch)
            loss = outputs.loss

            loss_buf += loss.item()
            train_losses.append(loss.item())
            loss.backward()

            if step % optimizer_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                print(
                    f'optimizers updated, step {completed_steps} / {epoch_num * len(train_dataloader)}, loss {loss_buf}'
                )
                loss_buf = 0

        print('model evaluating')
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            batch = batch.to(model.device)
            with torch.no_grad():
                outputs = model.forward(**batch)
                loss = outputs.loss

                eval_loss_sum += loss.item()

        for step, batch in enumerate(subset_dataloader):
            batch = batch.to(model.device)
            with torch.no_grad():
                outputs = model.forward(**batch)
                loss = outputs.loss

                train_loss_sum += loss.item()

        train_losses_epoch.append(train_loss_sum)
        eval_losses_epoch.append(eval_loss_sum)

        print(f'epoch {epoch + 1} / {epoch_num} completed, train_loss: {train_loss_sum}, eval_loss: {eval_loss_sum}')

    torch.save(model, 'content/model.zip')

    with open('content/train_losses_epoch.pickle', 'wb') as f:
        pickle.dump(train_losses_epoch, f)

    with open('content/eval_losses_epoch.pickle', 'wb') as f:
        pickle.dump(eval_losses_epoch, f)

    with open('content/train_losses.pickle', 'wb') as f:
        pickle.dump(train_losses, f)

    metric = load_metric('rouge')

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    gen_kwargs = {
        'max_length': max_source_length,
        'num_beams': num_beams
    }

    print('testing model')
    for step, batch in enumerate(test_dataloader):
        batch = batch.to(model.device)
        with torch.no_grad():
            bsz = batch['input_ids'].shape[0]
            past_prompt = model.get_prompt(bsz=bsz, sample_size=gen_kwargs['num_beams'])
            generated_tokens = model.seq2seq_model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                past_prompt=past_prompt,
                use_cache=True,
                **gen_kwargs
            )

            labels = batch['labels']

            labels = labels.cpu().numpy()
            generated_tokens = generated_tokens.cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            print(f'step {step + 1} / {len(test_dataloader)} completed')

    test_result = metric.compute(use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 4) for key, value in test_result.items()}

    print(result)


if __name__ == '__main__':
    main()
