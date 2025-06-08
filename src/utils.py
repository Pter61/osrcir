import json
import os
from typing import Optional, Tuple, List, Dict, Union

import argparse
import clip
import numpy as np
import openai_api
import pickle
import torch
import tqdm
# import cloudgpt_api


import data_utils
import prompts
from torch.cuda.amp import autocast
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


@torch.no_grad()
def extract_image_features(device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: clip.model.CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 8, preload: str=None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']        
        aux_data = {} if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

        index_features, index_names, index_ranks, aux_data = [], [], [], []
        if 'genecis' in args.dataset:
            aux_data = {'ref_features': [], 'instruct_features': []}
            
        try:
            print(f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features    
        index_rank = None
        for batch in tqdm.tqdm(loader):
            if 'genecis' in args.dataset:
                _, n_gallery, _, h, w = batch[3].size()
                images = batch[3].view(-1, 3, h, w)
                names, index_rank = batch[1], batch[4]
                ref_images = batch[0]
                instructions = batch[1]
            else:
                images = batch.get('image')
                names = batch.get('image_name')
                if images is None: images = batch.get('reference_image')
                if names is None: names = batch.get('reference_name')

            images = images.to(device)
            with torch.no_grad(),torch.cuda.amp.autocast():
                batch_features = clip_model.encode_image(images)
                index_features.append(batch_features.cpu())
                index_names.extend(names)
                if index_rank is not None:
                    index_ranks.extend(index_rank)
                if len(aux_data):
                    aux_data['ref_features'].append(clip_model.encode_image(ref_images.to(device)).cpu())
                    if hasattr(clip_model, 'tokenizer'):
                        aux_data['instruct_features'].append(clip_model.encode_text(clip_model.tokenizer(instructions, context_length=77).to(device)).cpu())
                    else:
                        aux_data['instruct_features'].append(clip_model.encode_text(clip.tokenize(instructions, context_length=77).to(device)).cpu())
        
        index_features = torch.vstack(index_features)
        
        if 'genecis' in args.dataset:
            # Reshape features into gallery-set for GeneCIS-style problems.
            index_features = index_features.view(-1, n_gallery, batch_features.size()[-1])
            index_ranks = torch.stack(index_ranks)
            aux_data['ref_features'] = torch.vstack(aux_data['ref_features'])
            aux_data['instruct_features'] = torch.vstack(aux_data['instruct_features'])
            
        if preload is not None:
            pickle.dump({'index_features': index_features, 'index_names': index_names, 'index_ranks': index_ranks, 'aux_data': aux_data}, open(preload, 'wb'))
    
    return index_features, index_names, index_ranks, aux_data


@torch.no_grad()
def OSrCIR(device: torch.device, args: argparse.Namespace, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]], **kwargs) -> Tuple[torch.Tensor, List[str], list]:
    print(preload_dict['mods'])
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        all_captions, all_relative_captions, all_modified_captions = [], [], []
        all_thoughts, all_reflations = [], []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []
        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=args.batch_size, num_workers=8,
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)

        query_iterator = tqdm.tqdm(query_loader, position=0, desc='Predicting Target captions with MLLM...')
        relative_captions = []
        for batch in query_iterator:
            if 'genecis' in args.dataset:
                # blip_image = batch[2].to(device)
                ref_image_path = batch[0]
                relative_captions.extend(batch[1])
            else:
                ref_image_path = batch['reference_image_path']
                reference_names.extend(batch['reference_name'])
                # print(ref_image_path)
                if 'fashioniq' not in args.dataset:
                    # relative_captions.extend(batch['relative_caption'])
                    relative_captions = batch['relative_caption']
                else:
                    rel_caps = batch['relative_captions']
                    rel_caps = np.array(rel_caps).T.flatten().tolist()

                    relative_captions = [
                        f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)]

                if 'target_name' in batch:
                    target_names.extend(batch['target_name'])

                gt_key = 'gt_img_ids'
                if 'group_members' in batch:
                    gt_key = 'group_members'
                if gt_key in batch:
                    gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

                query_key = 'query_id'
                if 'pair_id' in batch:
                    query_key = 'pair_id'
                if query_key in batch:
                    query_ids.extend(batch[query_key])
            query_iterator.set_postfix_str(f'Shape: {len(ref_image_path)}')
            sys_prompt = eval(args.gpt_cir_prompt)
            # print(sys_prompt)
            modified_captions = []
            thoughts, reflations = [], []
            for i in tqdm.trange(len(ref_image_path), position=1, desc='Iterating over batch', leave=False):
                instruction = relative_captions[i]

                user_prompt = '''
                <Input>
                    {
                        "Original Image": <image_url>
                        "Manipulation text": %s.
                    }
                ''' % instruction


                if isinstance(ref_image_path[i], str):
                    image_path = r"%s" % ref_image_path[i]

                # print(sys_prompt)
                # resp is a string in json format
                resp = cloudgpt_api.openai_completion_vision_CoT(sys_prompt=sys_prompt, user_prompt=user_prompt, image = image_path, engine=args.openai_engine)

                # Remove <Response> tags if present
                if resp.startswith('<Response>'):
                    resp = resp.replace('<Response>', '').replace('</Response>', '').strip()

                # Remove json tags if present
                if resp.startswith('```json'):
                    resp = resp.replace('```json', '').replace('```', '').strip()

                ## extract target image description
                # json.loads(resp) have the error
                try :
                    resp_dict = json.loads(resp)
                except:
                    modified_captions.append(relative_captions[i])
                    thoughts.append("")
                    reflations.append("")
                    continue

                # print(resp_dict.values())
                description = ""
                aug = False
                if 'Thoughts' in resp_dict:
                    description = resp_dict['Thoughts']
                    # print(description)
                    if description == "":
                        thoughts.append(relative_captions[i])
                    else:
                        thoughts.append(description)
                description = ""
                if 'Reflections' in resp_dict:
                    description = resp_dict['Reflections']
                    # print(description)
                    if description == "":
                        reflations.append(relative_captions[i])
                    else:
                        reflations.append(description)
                description = ""
                aug = False
                if 'Target Image Description' in resp_dict:
                    description = resp_dict['Target Image Description']
                    # print(description)
                    if description == "":
                        modified_captions.append(relative_captions[i])
                    else:
                        modified_captions.append(description)
                        aug = True
                if not aug:
                    modified_captions.append(relative_captions[i])

                print("Target Description: ", modified_captions[i])

            all_modified_captions.extend(modified_captions)
            all_thoughts.extend(thoughts)
            all_reflations.extend(reflations)
            all_relative_captions.extend(relative_captions)

        if preload_dict['mods'] is not None:

            res_dict = {
                'target_names': target_names,
                'targets': gt_img_ids,
                'reference_names': reference_names,
                'query_ids': query_ids,
                'start_captions': all_captions,
                'thoughts': all_thoughts,
                'reflections': all_reflations,
                'modified_captions': all_modified_captions,
                'instructions': all_relative_captions
            }
            pickle.dump(res_dict, open(preload_dict['mods'], 'wb'))
            print("\n", len(res_dict['target_names']), len(res_dict['reference_names']), len(res_dict['modified_captions']), len(res_dict['instructions']))
    else:
        print(f'Loading predicted target image captions from {preload_dict["mods"]}!')
        res_dict = pickle.load(open(preload_dict['mods'], 'rb'))
        target_names, gt_img_ids, reference_names, query_ids, all_captions, all_thoughts, all_reflations, all_modified_captions, all_relative_captions = res_dict.values()
        # print(target_names[1], reference_names[1], modified_captions[1], relative_captions[1])
    return target_names, gt_img_ids, reference_names, query_ids, all_captions, all_thoughts, all_reflations, all_modified_captions, all_relative_captions

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
def get_recall(indices, targets): #recall --> wether next item in session is within top K recommended items or not
    """
    Code adapted from: https://github.com/hungthanhpham94/GRU4REC-pytorch/blob/master/lib/metric.py
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B) or (BxN): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """

    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:        
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):            
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))        
        return torch.Tensor(recall).float().mean()
            
@torch.no_grad()            
def evaluate_genecis(device: torch.device, args: argparse.Namespace, clip_model: clip.model.CLIP, blip_model: callable, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]], topk: List[int] = [1,2,3], batch_size: int=32, **kwargs):
    val_loader = torch.utils.data.DataLoader(
        dataset=query_dataset, batch_size=batch_size, num_workers=8, 
        pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)            
    query_iterator = tqdm.tqdm(val_loader, position=0, desc='Generating image captions...')

    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []
    
    with torch.no_grad():
        for batch in query_iterator:
            ref_img = batch[0].to(device)
            original_caption = batch[1]
            caption = clip.tokenize(batch[1],context_length=77).to(device)
            blip_ref_img = batch[2].to(device)
            gallery_set = batch[3].to(device)
            target_rank = batch[4].to(device)

            bsz, n_gallery, _, h, w = gallery_set.size()
            imgs_ = torch.cat([ref_img,gallery_set.view(-1,3,h,w)],dim=0)
            
            # CLIP Encoding
            all_img_feats = clip_model.encode_image(imgs_).float()
            caption_feats = clip_model.encode_text(caption).float()

            # BLIP Captioning
            captions = []
            for i in tqdm.trange(bsz, position=1, desc=f'Captioning image with BLIP', leave=False):
                caption = blip_model.generate({"image": blip_ref_img[i].unsqueeze(0), "prompt": prompts.blip_prompt})
                captions.append(caption[0])
            
            modified_captions = []
            base_prompt = eval(args.llm_prompt)

            # LLM Caption Updates
            for i in tqdm.trange(len(captions), position=1, desc=f'Modifying captions with LLM', leave=False):
                instruction = original_caption[i]
                img_caption = captions[i]
                final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
                final_prompt = final_prompt + '\n' + 'Instruction: '+ instruction
                resp = openai_api.openai_completion(final_prompt)

                resp = resp.split('\n')

                description = ""
                for line in resp:                        
                    if line.startswith('Edited Description:'):
                        description = line.split(':')[1].strip()
                        modified_captions.append(description)
                        break
                if description == "":
                    modified_captions.append(original_caption[i])

            predicted_feature = torch.nn.functional.normalize(clip_model.encode_text(clip.tokenize(modified_captions,context_length=77).to(device)))
            
            ##### COMPUTE RECALL - Base Evaluation.
            ref_feats, gallery_feats = all_img_feats.split((bsz,bsz*n_gallery),dim=0)
            gallery_feats = gallery_feats.view(bsz,n_gallery,-1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            #combined_feats = F.normalize(ref_feats + caption_feats)
            combined_feats = predicted_feature
            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:
                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)
            sims_to_save.append(similarities.cpu())

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters
    
    
def text_encoding(device, clip_model, input_captions, batch_size=32, mode='default'):
    n_iter = int(np.ceil(len(input_captions)/batch_size))
    predicted_features = []
        
    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        
        if hasattr(clip_model, 'tokenizer'):
            tokenized_input_captions = clip_model.tokenizer(captions_to_use, context_length=77).to(device)
        else:
            tokenized_input_captions = clip.tokenize(captions_to_use, context_length=77, truncate=True).to(device)
        # input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        #clip_text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)        
        
    return torch.nn.functional.normalize(predicted_features, dim=-1)
