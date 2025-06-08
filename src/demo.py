import argparse
import prompts
import os
import pickle
import json
import tqdm
import cloudgpt_api
import torch
import datasets
import compute_results
import termcolor
from typing import Optional, Tuple, List, Dict, Union
import data_utils
import numpy as np
import json
import PIL

def parser_args():
    parser = argparse.ArgumentParser('')

    parser.add_argument("--preload", nargs='+', type=str, default=['img_features','captions','mods'],
                        help='List of properties to preload is computed once before.')
    parser.add_argument("--preload_path", nargs='+', type=str, default=r"C:\Users\v-yuantang\OneDrive\DKI\chatcir",
                        help='preload file path.')
    # Base Model Choices
    parser.add_argument("--clip", type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32','ViT-B-16','ViT-L-14','ViT-H-14','ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use"),
    parser.add_argument("--blip", type=str, default='blip2_t5', choices=['blip2_t5'],
                        help="BLIP Image Caption Model to use.")
    # Dataset Arguments ['dress', 'toptee', 'shirt']
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset")
    # LLM & BLIP Prompt Arguments.
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--gpt_cir_prompt", default='prompts.mllm_structural_modifier_prompt_fashion', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use alongside GPT4V. Has to be available in prompts.py')
    parser.add_argument("--openai_engine", default='gpt-35-turbo-1106', type=str,
                        choices=[   "gpt-35-turbo-20220309",
                                    "gpt-35-turbo-16k-20230613",
                                    "gpt-35-turbo-20230613",
                                    "gpt-35-turbo-1106",
                                    "gpt-4-20230321",
                                    "gpt-4-20230613",
                                    "gpt-4-32k-20230321",
                                    "gpt-4-32k-20230613",
                                    "gpt-4-1106-preview",
                                    "gpt-4-0125-preview",
                                    "gpt-4-visual-preview",
                                    "gpt-4-turbo-20240409",
                                    "gpt-4o-20240513",
                                    "gpt-4o-20240806",
                                    "gpt-4o-mini-20240718",],
                        help='Openai LLM Engine to use.')

    parser.add_argument("--batch_size", default=32, type=int,
                        help='Batch size to use.')
    args = parser.parse_args()
    return args

args = parser_args()

def get_predeal_dict():
    ### Argument Checks.
    preload_dict = {key: None for key in ['img_features', 'captions', 'mods']}
    preload_str = f'{args.dataset}_{args.openai_engine}_{args.clip}_{args.split}'.replace('/', '-') # fashioniq_dress_blip2_t5_ViT-g-14_val
    print(preload_str)

    if len(args.preload):
        os.makedirs(os.path.join(args.preload_path, 'precomputed'), exist_ok=True)
    if 'img_features' in args.preload:
        # # CLIP embeddings only have to be computed when CLIP model changes.
        # img_features_load_str = f'{args.dataset}_{args.clip}_{args.split}'.replace('/', '-')
        preload_dict['img_features'] = os.path.join(args.preload_path, 'precomputed', preload_str + '_img_features.pkl')

    if 'captions' in args.preload:
        # # BLIP captions only have to be computed when BLIP model or BLIP prompt changes.
        caption_load_str = f'{args.dataset}_{args.openai_engine}_{args.split}'.replace('/', '-')
        if args.gpt_cap_prompt != 'prompts.blip_prompt':
            preload_dict['captions'] = os.path.join(args.preload_path, 'precomputed',
                                                    caption_load_str + f'_captions_{args.gpt_cap_prompt.split(".")[-1]}.pkl')
        else:
            preload_dict['captions'] = os.path.join(args.preload_path, 'precomputed', caption_load_str + '_captions.pkl')

    if 'mods' in args.preload:
        # # LLM-based caption modifications have to be queried only when BLIP model or BLIP prompt changes.
        mod_load_str = f'{args.dataset}_{args.split}'.replace('/', '-')
        preload_dict['mods'] = os.path.join(args.preload_path, 'precomputed',
                                            mod_load_str + f'_mods_{args.gpt_cir_prompt.split(".")[-1]}.pkl')
        if args.openai_engine != 'gpt-3.5-turbo':
            preload_dict['mods'] = preload_dict['mods'].replace('.pkl', f'_{args.openai_engine}.pkl')

    if args.split == 'test':
        preload_dict[
            'test'] = preload_str + f'{args.blip_prompt.split(".")[-1]}_{args.llm_prompt.split(".")[-1]}_test_submission.json'
    return preload_dict

def OSrCIR(device: torch.device, args: argparse.Namespace, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str,None]], **kwargs) -> Tuple[torch.Tensor, List[str], list]:
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        all_captions, all_relative_captions, all_modified_captions = [], [], []
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


            query_iterator.set_postfix_str(f'Shape: {len(ref_image_path)}')
            sys_prompt = eval(args.gpt_cir_prompt)

            target_image_description = []
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
                    target_image_description.append(relative_captions[i])
                    continue

                aug = False
                if 'Target Image Description' in resp_dict:
                    description = resp_dict['Target Image Description']
                    # print(description)
                    if description == "":
                        target_image_description.append(relative_captions[i])
                    else:
                        target_image_description.append(description)
                        aug = True
                if not aug:
                    target_image_description.append(relative_captions[i])
                # print("Target Description: ", modified_captions[i])
                # print("Target Description: ", modified_captions[i], "Target name: ", target_names[i], "Reference name: ", reference_names[i], "Instruction: ", instruction)
                # print(modified_captions)
                print("Target Description: ", target_image_description[i])

if __name__ == "__main__":
    # --- Set Device.
    termcolor.cprint(f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # --- Load Evaluation Datasets.
    target_datasets, query_datasets, pairings = [], [], []
    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'classic'))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [dress_type], 'relative'))
        pairings.append(dress_type)
        compute_results_function = compute_results.fiq

    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'classic'))
        query_datasets.append(datasets.CIRRDataset(args.dataset_path, split, 'relative'))
        compute_results_function = compute_results.cirr
        pairings.append('default')

    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'classic'))
        query_datasets.append(datasets.CIRCODataset(args.dataset_path, args.split, 'relative'))
        compute_results_function = compute_results.circo
        pairings.append('default')

    elif 'genecis' in args.dataset.lower():
        data_split = '_'.join(args.dataset.lower().split('_')[1:])
        prop_file = os.path.join(args.dataset_path, 'genecis', data_split + '.json')

        if 'object' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
            genecis_dataset = datasets.COCOValSubset(root_dir=datapath, val_split_path=prop_file, data_split=data_split)
        elif 'attribute' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'Visual_Genome', 'VG_All')
            genecis_dataset = datasets.VAWValSubset(image_dir=datapath, val_split_path=prop_file, data_split=data_split)

        target_datasets.append(genecis_dataset)
        query_datasets.append(genecis_dataset)
        compute_results_function = compute_results.genecis
        pairings.append('default')

    # --- get predeal dicts from each stage
    preload_dict = get_predeal_dict()

    # --- Evaluate performances.
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])

        ### General Input Arguments.
        input_kwargs = {
            'device': device , 'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'preload_dict': preload_dict,
        }
        # --- Predict target captions
        OSrCIR(**input_kwargs)