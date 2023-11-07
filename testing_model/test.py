from transformers import XLMRobertaModel, XLMRobertaTokenizer
model = XLMRobertaModel.from_pretrained("bonadossou/afrolm_active_learning")
tokenizer = XLMRobertaTokenizer.from_pretrained("bonadossou/afrolm_active_learning")
tokenizer.model_max_length = 256
