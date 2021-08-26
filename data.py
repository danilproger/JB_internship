from datasets import load_dataset


def get_cnn_dataset():
    return load_dataset('cnn_dailymail', '3.0.0')


def process_dataset(dataset, tokenizer, max_source_length, max_target_length, padding, batch_size=1, remove_columns=None):
    def process_data_to_model_inputs(example):
        inputs = example['article']
        targets = example['highlights']

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        labels['input_ids'] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in
                               labels['input_ids']]

        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    return dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=remove_columns,
        load_from_cache_file=True,
    )


def train_eval_test_split(dataset):
    return dataset['train'], dataset['validation'], dataset['test']


def get_subset(dataset, subset_size):
    return dataset.select(range(subset_size))
