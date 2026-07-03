
class SequenceBuilder:

    def __init__(self, tokenizer, sequence_length):

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def build(self, texts):

        inputs = []
        targets = []

        for text in texts:

            token_ids = self.tokenizer.encode(text)

            if len(token_ids) < 2:
                continue

            for i in range(len(token_ids) - 1):

                input_ids = token_ids[:i + 1]
                target_ids = token_ids[1:i + 2]

                if len(input_ids) > self.sequence_length:

                    input_ids = input_ids[-self.sequence_length:]
                    target_ids = target_ids[-self.sequence_length:]

                inputs.append(input_ids)
                targets.append(target_ids)

        return inputs, targets
