import os
import torch
import datetime
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
import torch.nn as nn
import random
import numpy as np

warnings.filterwarnings(action='ignore') 
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10, #Your Epochs,
    'LR':1e-5, #Your Learning Rate,
    'BATCH_SIZE':4, #Your Batch Size,
    'MAX_LENGTH':128,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device) 
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())  

import wandb

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = os.path.join("/mnt/datasets/Dacon_Image_Quality_Assessment/", self.dataframe.iloc[idx]['img_path']) # added os.path.join(~)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        
        return img, comment
# 데이터 로드
train_data = pd.read_csv("/mnt/datasets/Dacon_Image_Quality_Assessment/train.csv")

# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],std=[0.26862954,0.26130258,0.27577711])
])

##### HuggingFace BLIP2 #####

from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", do_convert_rgb=False, do_normalize=False, do_rescale=False, do_resize=False)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map='auto', torch_dtype=torch.float16)
processor.tokenizer.model_max_length = CFG['MAX_LENGTH']

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item[0], text=item[1], padding="max_length", return_tensors="pt") # edited
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

from peft import LoraConfig, get_peft_model
# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
    # target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)

for name, param in model.named_parameters():
    if 'qformer' in name or 'vision_model.encoder' in name or 'language_model.model.decoder.layers.31' in name:
        param.requires_grad = True
        
model.print_trainable_parameters()

val_data = train_data[70000:].reset_index(drop=True)
train_data = train_data[:70000]
# train_data = pd.read_csv('/home/6210seok/data/train_concat.csv')

train_dataset = CustomDataset(train_data, transform)
train_dataset = ImageCaptioningDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=CFG['BATCH_SIZE'], num_workers=4, collate_fn=collate_fn)

val_dataset = CustomDataset(val_data, transform)
val_dataset = ImageCaptioningDataset(val_dataset, processor)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=CFG['BATCH_SIZE'], num_workers=4, collate_fn=collate_fn)

test_data = pd.read_csv("/home/6210seok/data/test.csv")
test_dataset = CustomDataset(test_data, transform)
test_dataset = ImageCaptioningDataset(test_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=CFG['BATCH_SIZE'], num_workers=4, collate_fn=collate_fn)

import nltk
from nltk import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def val(model, val_dataloader):
    model.eval()
    val_loss = 0
    meteor, bleu4, bleu3, rougeL = [], [], [], []
    i=1
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader)):
            val_predicted_comments_list = []
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").float().to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)

            generated_ids = model.generate(pixel_values=pixel_values, max_length=CFG['MAX_LENGTH'])
            val_predicted_comments_list += processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            loss = outputs.loss
            val_loss += loss.item()
            wandb.log({"val_loss": val_loss / i})
            
            for j in range(len(val_predicted_comments_list)):
                reference, candidate = val_data['comments'][(i-1)*CFG['BATCH_SIZE'] + j], val_predicted_comments_list[j]
                
                meteor.append(meteor_score([word_tokenize(reference)], word_tokenize(candidate)))
                bleu4.append(sentence_bleu([reference.split()], candidate.split(), weights=(0.25,0.25,0.25,0.25)))
                bleu3.append(sentence_bleu([reference.split()], candidate.split(), weights=(1/3.,1/3.,1/3.)))
                rougeL.append(scorer.score(reference, candidate)['rougeL'])
            i+=1
        print(f"Val loss: {val_loss / len(val_dataloader):.4f}")
        meteor, bleu4, bleu3, rougeL = np.array(meteor).mean(), np.array(bleu4).mean(), np.array(bleu3).mean(), np.array(rougeL).mean()
        
        # wandb score log
        wandb.log({"meteor": meteor})
        wandb.log({"bleu4": bleu4})
        wandb.log({"bleu3": bleu3})
        wandb.log({"rougeL": rougeL})
        wandb.log({"score": 3*meteor + bleu4 + bleu3 + rougeL})
        
        
        
def train(model, optimizer, train_dataloader, val_dataloader=None):
    
    wandb.init(project="dacon-image-captioning", name="BLIP2-2.7b", notes=str(torch.cuda.get_device_name())+' x '+str(1), config=CFG)
    
    for epoch in range(CFG['EPOCHS']):
        model.train()
        total_loss = 0
        
        loop = tqdm(enumerate(train_dataloader), leave=True, total=len(train_data)//CFG['BATCH_SIZE'])
        for idx, batch in loop:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").float().to(torch.float16).to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)
    
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
            wandb.log({'Epoch': epoch+1, "during_train_loss": loss})
            
        print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_dataloader):.4f}")
        
        
        if val_dataloader != None:
            val(model, val_dataloader)

        if epoch == 4:
            torch.save(model.state_dict(), 'BLIP2-'+str(epoch+1)+'.pt')
        
# test
test_data = pd.read_csv("/home/6210seok/data/test.csv")
def test(model, test_dataloader):
    model.eval()

    predicted_comments_list = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_dataloader)):
            pixel_values = batch.pop("pixel_values").float().to(torch.float16).to(device)

            generated_ids = model.generate(pixel_values=pixel_values, max_length=CFG['MAX_LENGTH'])
            
            predicted_comments_list += processor.batch_decode(generated_ids, skip_special_tokens=True)

    result_df = pd.DataFrame({
        'img_name': test_data['img_name'],
        'mos': 0, # default value
        'comments': predicted_comments_list
    })

    result_df.to_csv('BLIP2-2.7b.csv', index=False)


all_parameters_except_text_decoder = [param for name, param in model.named_parameters() if "text_decoder" not in name]
optimizer = torch.optim.AdamW([{'params': model.text_decoder.parameters(), 'lr':CFG['LR']/10},
                               {'params': all_parameters_except_text_decoder, 'lr': CFG['LR']}])

train(model, optimizer, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'BLIP2-2.7b.pt')

test(model, test_dataloader)