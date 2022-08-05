from transformers import AutoConfig, AutoModelForSequenceClassification

from proselflc.exceptions import ParamException


def prot_bert_bfd_seqclassifier(
    params: dict = {
        "network_name": "Rostlab/prot_bert_bfd",
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        #
        "num_classes": 2,
    }
):
    if params["network_name"] == "Rostlab_prot_bert_bfd_seq":
        network_name = "Rostlab/prot_bert_bfd"
        config = AutoConfig.from_pretrained(network_name)
        config.num_hidden_layers = params["num_hidden_layers"]
        config.num_attention_heads = params["num_attention_heads"]

        config.num_labels = params["num_classes"]

        return AutoModelForSequenceClassification.from_pretrained(
            network_name,
            config=config,
        )
    else:
        error_msg = "network_name != Rostlab_prot_bert_bfd_seq"
        raise ParamException(error_msg)
