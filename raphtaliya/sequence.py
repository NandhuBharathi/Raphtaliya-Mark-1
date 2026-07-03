
class SequenceBuilder:

    def __init__(self, tokenizer, sequence_length=32):

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def build(self, texts):

        inputs = []
        targets = []

        for text in texts:

            token_ids = self.tokenizer.encode(text)

            if len(token_ids) <= self.sequence_length:
                continue

            for i in range(len(token_ids) - self.sequence_length):

                input_ids = token_ids[
                    i:i+self.sequence_length
                ]

                target_ids = token_ids[
                    i+1:i+self.sequence_length+1
                ]

                inputs.append(input_ids)
                targets.append(target_ids)

        return inputs, targets
