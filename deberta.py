from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/deberta-v3-large"
save_dir = "artifacts/model/deberta-v3-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)