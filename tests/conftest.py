from __future__ import annotations

import pytest
import torch

from actlogit.config import Config, ModelConfig, PromptConfig
from actlogit.schema import Choice, ChoiceQuestion, DecisionRequest


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(12)
    path = tmp_path_factory.mktemp("tiny-local-model")
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "A",
        "B",
        "C",
        "D",
        "E",
        "0",
        "1",
        "2",
        "Select",
        "Choose",
        "the",
        "best",
        "choice",
        "state",
        "instructions",
        "choices",
        "label",
        "id",
        "description",
        "payload",
        "billing",
        "technical",
        "account",
        "ticket",
        "refund",
        "bug",
        "password",
        "Choice",
        ":",
        ".",
        ",",
        '"',
        "{",
        "}",
    ]
    tokenizer_impl = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, "[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(words),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(path)
    return path


@pytest.fixture
def config(tiny_model):
    return Config(
        model=ModelConfig(name_or_path=str(tiny_model), device="cpu", dtype="float32"),
        prompt=PromptConfig(format="plain", labels=["A", "B", "C", "D", "E"]),
    )


@pytest.fixture
def decision():
    return DecisionRequest(
        state={"ticket": "refund"},
        instructions="Choose the best team.",
        choices=[
            Choice(id="billing", description="refund"),
            Choice(id="technical", description="bug"),
            Choice(id="account", description="password"),
        ],
    )


@pytest.fixture
def choice_question(decision):
    return ChoiceQuestion(
        instructions=decision.instructions,
        criteria={choice.id: choice.description for choice in decision.choices},
    )
