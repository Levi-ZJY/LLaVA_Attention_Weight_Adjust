import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)


        # 已证实，每次输入仅一个image和相关system prompt以及question
        # print("input_ids:", input_ids)
        # print("image_tensor:", image_tensor.size())
        # print("image_sizes:", image_sizes)

        """
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        """

        """!!!"""  # 输入image和question

        # print("input_ids", input_ids, input_ids.size())
        
        with torch.inference_mode():
            generate_output  = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,   # 返回字典
                output_scores=True              # 返回每一步的logits分数
            )

        output_ids = generate_output.sequences   

        logits_list = generate_output.scores

        """
        第 1 个 token 是 <bos>（ID = 1）：
        这是在解码开始前就预先放进输入序列的，模型不会为这个 <bos> 再输出一个 logits 步骤。也就是说，它并非 “生成出来” 的，而是作为输入 prompt 的一部分存在
        
        解码实际只发生了 2 步：
            Step 1：<bos> → 1939
            Step 2：1939 → <eos>
        这也就对应了 logits_list 里只有 2 个元素
        """
        
        print("############output_ids:", output_ids)   # output_ids: tensor([[1, 1939, 2]])  1是开始标记，2是结束标记
        
        for i, tensor in enumerate(logits_list):
            print(f"Logit {i} shape: {tensor.shape}")
            max_value, max_index = torch.max(tensor, dim=1)

            # 经验证，这里使用的应该是greedy算法
            # print("max_index", max_index.item())
            # print("max_value:", max_value.item())
        
        """!!!"""






        
        """!!!""" # 仅输入question

        input_ids_no_img = input_ids[input_ids != -200].view(1, -1)
        
        # print("input_ids_no_img", input_ids_no_img, input_ids_no_img.size())
        
        with torch.inference_mode():
            generate_output_no_image  = model.generate(
                input_ids_no_img,
                images=None,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,   # 返回字典
                output_scores=True              # 返回每一步的logits分数
            )
            
        output_ids_no_image = generate_output_no_image.sequences   

        logits_list_no_image = generate_output_no_image.scores

        print("############output_ids_no_image:", output_ids_no_image)

        for i, tensor in enumerate(logits_list_no_image):
            print(f"Logit {i} shape: {tensor.shape}")

        
        print("------------------------------------------------------")
        if len(logits_list) == len(logits_list_no_image):
            print(len(logits_list))
            for i in range(len(logits_list)):
                token_idx = output_ids[0][i+1]
                token_idx_no_img = output_ids_no_image[0][i+1]
                
                # logit shape: ( torch.Size([1, 32000]), torch.Size([1, 32000]), torch.Size([1, 32000]), ... )
                logit = logits_list[i][0][token_idx]
                logit_no_img = logits_list_no_image[i][0][token_idx]
                delta_logit = logit - logit_no_img
                
                if output_ids[0][i+1] == output_ids_no_image[0][i+1]:
                    print(f"PredictedToken{i}: [Same] logit->{logit} | logit_no_img->{logit_no_img} | delta_logit->{delta_logit} | with: {output_ids[0][i+1]} | without: {output_ids_no_image[0][i+1]} |")
                else:
                    print(f"PredictedToken{i}: [Diff] logit->{logit} | logit_no_img->{logit_no_img} | delta_logit->{delta_logit} | with: {output_ids[0][i+1]} | without: {output_ids_no_image[0][i+1]}->{logits_list_no_image[i][0][token_idx_no_img]}|")
        
        else:
            print("Not the same length!")

        print("------------------------------------------------------")
        
        print("!!!")
        
        
        """!!!"""














        
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
