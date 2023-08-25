import os
from abc import ABC, abstractmethod
from typing import List, Dict, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Modify this to change cache location
# os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/JawTitan/huggingface/")

# Acknowledgements:
# Modified from https://github.com/evalplus/evalplus/blob/master/codegen/model.py


HUMANEVAL_EOS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
NON_CODE_EOS = ["<|endoftext|>", "\n```", "\n</s>", "<|endofmask|>"]
EOS = HUMANEVAL_EOS + NON_CODE_EOS


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def perplexity(self, prompt: str, completion: str) -> float:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class HFTorchDecoder(DecoderBase):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ):
        super().__init__(
            name=name, batch_size=batch_size, temperature=temperature
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        kwargs: Dict[str, Union[str, bool, torch.dtype]] = {
            "trust_remote_code": name
            in {
                "bigcode/santacoder",
                "Salesforce/codegen2-1B",
                "Salesforce/codegen2-3_7B",
                "Salesforce/codegen2-7B",
                "Salesforce/codegen2-16B",
                "mosaicml/mpt-30b",
                "mosaicml/mpt-7b-instruct",
                "tiiuae/falcon-7b",
                "tiiuae/falcon-7b-instruct",
            }
        }
        if "codegen-" in name:  # use fp16 for codegen models
            kwargs["torch_dtype"] = torch.float16
        if "codegen2-" in name:  # avoid warning of trust remote code
            kwargs["revision"] = "main"
            kwargs["torch_dtype"] = torch.float16
            if "16b" in name.lower():
                kwargs["device_map"] = "auto"
                # Not working... # int8 acceleration
                # kwargs["load_in_8bit"] = True
        if "starcoder" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "WizardCoder" in name:  # use fp16 for wizardcoder models
            kwargs["torch_dtype"] = torch.float16
        if "Platypus" in name:  # use fp16 for stable-platypus2 models
            kwargs["torch_dtype"] = torch.float16
        if "mpt-" in name:  # use fp16 for mpt models
            config = AutoConfig.from_pretrained(name, trust_remote_code=True)
            # config.attn_config['attn_impl'] = 'triton'
            config.init_device = "cuda"
            kwargs["config"] = config
            kwargs["torch_dtype"] = torch.bfloat16
        if "falcon" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "StableBeluga" in name:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        if "WizardCoder" in name:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model = self.model.half()
        if "Platypus" in name:
            self.model = self.model.half()
        if name in {"StabilityAI/stablelm-base-alpha-7b"}:
            print("Switching to float16 ...")
            self.model = self.model.half()
            self.skip_special_tokens = True
        self.model = self.model.to(self.device)

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(
            self.device
        )
        completion_tokens = self.tokenizer(
            prompt + completion, return_tensors="pt"
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    # Assumption is that all inputs should probably fit under maximum context. but can add a checking function
    # just in case. TODO: think about
    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class IncoderDecoder(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.infill_ph = "<|mask:0|>"
        self.extra_end = "<|mask:1|><|mask:0|>"
        self.extra_eos = [
            "<|endofmask|>",
            "<|/ file",
            "</cell>",
            "</text>",
            "</code>",
            "<|",
            "</CODE>",
        ]
        self.eos = self.eos + self.extra_eos

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            prompt + self.infill_ph + self.extra_end, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            prompt + self.infill_ph + self.extra_end + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class Codegen2Decoder(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.infill_ph = "<mask_1>"
        # taken from: https://huggingface.co/Salesforce/codegen2-16B
        self.extra_end = "<|endoftext|><sep><mask_1>"
        self.extra_eos = ["<eom>"]
        self.eos = self.eos + self.extra_eos

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class SantaCoder(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = "<fim-prefix>"
        self.suffix_token = "<fim-suffix>\n<fim-middle>"
        self.extra_eos = ["<|endofmask|>"]
        self.eos = self.eos + self.extra_eos

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs,
            skip_special_tokens=self.skip_special_tokens,
            truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class StarCoder(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        temperature = max(self.temperature, 1e-2)
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class WizardCoder(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = (
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request.\n\n### Instruction:\n "
        )
        self.suffix_token = "\n### Response:"

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class StablePlatypus2(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = "### Instruction:\n"
        self.suffix_token = "\n### Response:"

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class MPTInstruct(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n"
        self.suffix_token = "\n### Response:"

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class StableBeluga(HFTorchDecoder):
    def __init__(
        self, name: str, batch_size: int = 1, temperature: float = 0.8
    ) -> None:
        super().__init__(name, batch_size, temperature)
        self.prefix_token = "### System:\nYou are StableBeluga, an AI that follows instructions extremely well. Help " \
                            "as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\n "
        self.suffix_token = "\n### Assistant:"

    @torch.inference_mode()
    def perplexity(self, prompt: str, completion: str) -> float:
        input_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token, return_tensors="pt"
        ).to(self.device)
        completion_tokens = self.tokenizer(
            self.prefix_token + prompt + self.suffix_token + completion,
            return_tensors="pt",
        ).to(self.device)

        begin_loc = input_tokens.input_ids.size(1)
        input_ids = completion_tokens.input_ids
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        return ppl

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs
