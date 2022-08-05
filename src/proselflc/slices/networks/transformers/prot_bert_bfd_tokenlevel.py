from transformers import AutoConfig, AutoModelForTokenClassification

from proselflc.exceptions import ParamException


# hard to decoupled due to datasets
# leave it now
def prot_bert_bfd_tokenclassifier(
    params: dict = {
        "network_name": "Rostlab/prot_bert_bfd",
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        #
        "num_classes": 3,
    }
):
    if params["network_name"] == "Rostlab_prot_bert_bfd_token":
        network_name = "Rostlab/prot_bert_bfd"
        config = AutoConfig.from_pretrained(network_name)
        config.num_hidden_layers = params["num_hidden_layers"]
        config.num_attention_heads = params["num_attention_heads"]

        config.num_labels = params["num_classes"]

        return AutoModelForTokenClassification.from_pretrained(
            network_name,
            config=config,
        )
    else:
        error_msg = "network_name != Rostlab_prot_bert_bfd_token"
        raise ParamException(error_msg)

    # if "num_classes" not in params.keys():
    #    params["num_classes"] = len(unique_tags)
    # if "id2tag" not in params.keys():
    #    params["id2tag"] = id2tag
    # if "tag2id" not in params.keys():
    #    params["tag2id"] = tag2id

    # config = AutoConfig.from_pretrained(params["network_name"])
    # config.num_hidden_layers = params["num_hidden_layers"]
    # config.num_attention_heads = params["num_attention_heads"]

    # config.num_labels = params["num_classes"]
    # config.id2label = params["id2tag"]
    # config.label2id = params["tag2id"]

    # return AutoModelForTokenClassification.from_pretrained(
    #     params["network_name"],
    #     config=config,
    # )
