from warnings import warn

from llm_battle_ground.models.hugging_face.model import (
    Codegen2Decoder,
    HFTorchDecoder,
    IncoderDecoder,
    SantaCoder,
    StarCoder,
    WizardCoder,
    StablePlatypus2,
    MPTInstruct,
)


def make_model_hugging_face(
    name: str, batch_size: int = 1, temperature: float = 0.8
):
    if name == "codegen-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-2B-mono",
            temperature=temperature,
        )
    elif name == "codegen-6b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-6B-mono",
            temperature=temperature,
        )
    elif name == "codegen2-1b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-1B",
            temperature=temperature,
        )
    elif name == "codegen2-3b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-3_7B",
            temperature=temperature,
        )
    elif name == "codegen2-7b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-7B",
            temperature=temperature,
        )
    elif name == "codegen2-16b":
        warn(
            "codegen2-16b checkpoint is `unfinished` at this point (05/11/2023) according to their paper. "
            "So it might not make sense to use it."
        )
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-16B",
            temperature=temperature,
        )
    elif name == "polycoder":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="NinedayWang/PolyCoder-2.7B",
            temperature=temperature,
        )
    elif name == "santacoder":
        return SantaCoder(
            batch_size=batch_size,
            name="bigcode/santacoder",
            temperature=temperature,
        )
    elif name == "incoder-1b":
        return IncoderDecoder(
            batch_size=batch_size,
            name="facebook/incoder-1B",
            temperature=temperature,
        )
    elif name == "incoder-6b":
        return IncoderDecoder(
            batch_size=batch_size,
            name="facebook/incoder-6B",
            temperature=temperature,
        )
    elif name == "stablelm-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="StabilityAI/stablelm-base-alpha-7b",
            temperature=temperature,
        )
    elif name == "gptneo-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="EleutherAI/gpt-neo-2.7B",
            temperature=temperature,
        )
    elif name == "gpt-j":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="EleutherAI/gpt-j-6B",
            temperature=temperature,
        )
    elif name == "starcoder":
        return StarCoder(
            batch_size=batch_size,
            name="bigcode/starcoder",
            temperature=temperature,
        )
    elif name == "wizardcoder":
        return WizardCoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-15B-V1.0",
            temperature=temperature,
        )
    elif name == "platypus":
        return StablePlatypus2(
            batch_size=batch_size,
            name="garage-bAInd/Stable-Platypus2-13B",
            temperature=temperature,
        )
    elif name == "mpt":  # currently 40GB cannot support this model.
        return HFTorchDecoder(
            batch_size=batch_size,
            name="mosaicml/mpt-30b",
            temperature=temperature,
        )
    elif name == "mpt-instruct":
        return MPTInstruct(
            batch_size=batch_size,
            name="mosaicml/mpt-7b-instruct",
            temperature=temperature,
        )

    raise ValueError(f"Invalid model name: {name}")


def make_model(
    provider: str, name: str, batch_size: int = 1, temperature: float = 0.8
):
    if provider == "hugging-face":
        return make_model_hugging_face(
            name, batch_size=batch_size, temperature=temperature
        )
    else:
        raise NotImplementedError("No such provider.")
