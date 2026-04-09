
import os, sys, cv2, json, random
from PIL import Image
import numpy as np
from nltk.corpus import wordnet as wn
from TAM.tam_original.tam import TAM
import unicodedata
import string
import nltk
from tqdm import tqdm
from nltk.translate import meteor_score
from nltk.stem import WordNetLemmatizer
from rouge import Rouge
import warnings


# prepare
random.seed(1024)
warnings.filterwarnings("ignore")

if not os.path.exists(os.path.join(os.path.expanduser("~"), 'nltk_data/taggers/averaged_perceptron_tagger.zip')):
    nltk.download('averaged_perceptron_tagger')


def get_word_type(word):
    """
    Determine the general word type of a given word using NLTK's POS tagging.

    Args:
        word (str): A single word to be classified.

    Returns:
        str: A category label for the word type:
             - 'function' for function words (e.g., pronouns, determiners, conjunctions)
             - 'noun' for noun forms (singular, plural, proper)
             - 'others' for all other POS tags
             
    Note:
        Tags like 'IN' (prepositions) and 'CD' (cardinal numbers) are excluded from the 'function' category 
        for better readability.
    """

    tagged_word = nltk.pos_tag([word])
    pos = tagged_word[0][1]
    
    if pos in ['CC', 'DT', 'EX', 'MD', 'POS', 'PRP', 'PRP$', 'UH', 'WDT', 'WP', 'WP$', 'WRB']:
        return 'function'
    elif pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        return 'noun'
    else:
        return 'others'


def is_english_punctuation(char):
        return char in string.punctuation


def is_chinese_char_or_punctuation(char):
    for ch in char:
        if 'CJK' in unicodedata.name(ch, ''):
            return True
    return False


def ids_to_word_groups(ids, processor):
    """
    Decode token ids into grouped words and record the corresponding token indices.

    Args:
        ids (list[int]): List of token ids to decode.
        processor: Tokenizer processor with `batch_decode` and `tokenizer` attributes.

    Returns:
        tuple:
            - words (list[str]): List of decoded word groups.
            - tokens_idx (list[list[int]]): List of lists containing token indices for each word group.

    Notes:
        - Groups tokens together based on whitespace, punctuation, and special token markers (e.g., '▁').
        - Uses helper functions `is_english_punctuation` and `is_chinese_char_or_punctuation` to identify punctuation.
        - Removes spaces in decoded words to form continuous word strings.
    """

    txt = processor.batch_decode(ids)[0]
    tokens = processor.tokenizer.tokenize(txt)
    words, tokens_idx = [], []
    for i, _ in enumerate(tokens):
        word = processor.tokenizer.decode(processor.tokenizer.convert_tokens_to_ids(_))
        if i == 0 or is_english_punctuation(word) or is_chinese_char_or_punctuation(word) or word[0] == ' ' or _[0] == '▁':
            words.append(word.replace(' ', ''))
            tokens_idx.append([i])
        else:
            words[-1] += word.replace(' ', '')
            tokens_idx[-1].append(i)
    return words, tokens_idx


# match the same word (avoid to minus the same object)
lemmatizer = WordNetLemmatizer()
def single_words_match(word1, word2):
    a = lemmatizer.lemmatize(word1.lower().replace('-', ''))
    b = lemmatizer.lemmatize(word2.lower().replace('-', ''))
    return a == b


def words_match(category_word, target_word):
    """
    Check if any individual word in the category_word string matches the target_word.

    Args:
        category_word (str): A string containing one or more words separated by spaces.
        target_word (str): The word to match against.

    Returns:
        bool: True if any single word in category_word matches target_word according to 
              `single_words_match`, otherwise False.
    """

    tks = category_word.split()
    for tk in tks:
        if single_words_match(tk, target_word):
            return True
    return False


def resize(img, min_side):
    w, h  = img.size
    if w < h:
        w_ = min_side
        h_ = int(float(h) / w * w_)
    else:
        h_ = min_side
        w_ = int(float(w) / h * h_)
    return img.resize((w_, h_))


def evaluate(maps, tokens, processor, caption, mask, category):
    """
    Evaluate plausibility using Intersection over Union (IoU) between predicted maps and ground truth masks,
    and compute natural language generation (NLG) metrics for the generated captions.

    The function processes tokenized words, categorizes them as nouns, function words, or others,
    matches nouns to categories, and calculates IoU-based plausibility scores for nouns.
    It also computes plausibility for function words based on foreground thresholds derived from nouns.
    Finally, it computes ROUGE-L and METEOR scores comparing the generated captions to reference captions.

    Args:
        maps (List[np.ndarray]): List of predicted score maps corresponding to token segments.
        tokens (List[int]): Token ids representing the generated caption sequence.
        processor (object): Tokenizer/processor with methods to decode tokens into words.
        caption (List[str]): List of reference captions for evaluation.
        mask (str): File path to the ground truth segmentation mask image.
        category (dict): Dictionary mapping category names (str) to label ids (int).

    Returns:
        List: A list containing:
            - obj_iou (List[float]): IoU-based plausibility scores for noun objects.
            - func_iou (List[float]): Plausibility scores for function words based on foreground thresholds.
            - rougel (List[float]): ROUGE-L F1 scores comparing generated and reference captions.
            - meteor (List[float]): METEOR scores comparing generated and reference captions.
            - pre (List[float]): Precision values corresponding to noun tokens.
            - rec (List[float]): Recall values corresponding to noun tokens.

    Notes:
        - Nouns not found in the category dictionary are assigned a label of -1.
        - Function words are assigned a label of -2, others -3.
        - If the last token index does not match the number of maps minus one, returns empty results.
        - Thresholding uses Otsu's method on resized maps to binary masks.
        - Merges plausibility scores for consecutive words referring to the same object category.
    """

    # tokens to words to match category, noun and function words
    words, tokens_id = ids_to_word_groups(tokens, processor)
    if tokens_id[-1][-1] != (len(maps) - 1):
        return [[], [], [], [], [], []]
    words_label = []
    for word in words:
        word_type = get_word_type(word)
        if word_type == 'noun':
            lb = -1 # noun not in category
            for k, v in category.items():
                if words_match(k, word):
                    lb = v
            words_label.append(lb)
        elif word_type == 'function':
            words_label.append(-2) # function words
        else:
            words_label.append(-3) # other words

    # count the iou as object plausibility
    if os.path.exists(mask):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros_like(maps[0])
    obj_iou, pre, rec, noun_fg_thresh = [], [], [], []
    for i in range(len(words)):
        if words_label[i] > 0:
            ious, pres, recs, thresh = [], [], [], []
            gt = (mask == words_label[i]).astype('uint8')
            for j in tokens_id[i]:
                map = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, pred = cv2.threshold(map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if gt.sum() != 0:
                    tp = float((gt * pred > 0).sum())
                    ious.append(tp / ((gt + pred / 255) > 0).sum()) # /255 to avoid sum overflow
                    pres.append(tp / (pred > 0).sum())
                    recs.append(tp / (gt > 0).sum())
                thresh.append(t)

            # return the max iou and fg_thresh of tokens from a single word
            noun_fg_thresh.append(max(thresh))
            if len(ious) > 0:
                m_iou = max(ious)
                obj_iou.append(max(ious))
                pre.append(pres[ious.index(m_iou)])
                rec.append(recs[ious.index(m_iou)])

            # merge ious of multiple words that describe the same object (the same next word label)
            if len(obj_iou) > 1 and words_label[i] > 0 and words_label[i - 1] == words_label[i]:
                select_idx = -1 if obj_iou[-1] > obj_iou[-2] else -2
                obj_iou[-2] = obj_iou[select_idx]
                obj_iou = obj_iou[:-1]
                pre[-2] = pre[select_idx]
                pre = pre[:-1]
                rec[-2] = rec[select_idx]
                rec = rec[:-1]

        # also count fg_thresh of other noun words
        elif words_label[i] == -1:
            thresh = []
            for j in tokens_id[i]:
                map = cv2.resize(maps[j], (mask.shape[1], mask.shape[0]))
                t, pred = cv2.threshold(map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh.append(t)
            noun_fg_thresh.append(max(thresh))

    # count the iou of fundtion words, threshhold is the mean value of foreground of noun tokens
    func_iou = []
    if len(noun_fg_thresh) > 0:
        fg_thresh = sum(noun_fg_thresh) / len(noun_fg_thresh)
        for i in range(len(words)):
            if words_label[i] == -2:
                neg_iou = [] # neg gt->all, neg pred->lower than thresh
                for j in tokens_id[i]:
                    neg_iou.append(float((maps[j] < fg_thresh).sum()) / maps[j].size)
                func_iou.append(sum(neg_iou) / len(neg_iou))

    # compute nlg metrics
    output_text = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ref = [str(_).lower().split() for _ in caption]
    hypo = str(output_text[0]).lower().split()
    meteor = [meteor_score.meteor_score(references=ref, hypothesis=hypo)]
    r = Rouge()
    rougel = [max([r.get_scores(output_text[0], _)[0]['rouge-l']['f'] for _ in caption])]

    return [obj_iou, func_iou, rougel, meteor, pre, rec]


def eval_qwen2vl(model_name='Qwen/Qwen2-VL-2B-Instruct', input_data=[], vis_path='', reso=-1):
    """
    Evaluate the visual exaplainability of Qwen2-VL and visualize each token and example.

    Args:
        model_name (str): Pretrained model identifier or path.
        input_data (list): List of tuples containing (image or video, prompt text, caption, mask, category).
        vis_path (str): Directory path to save visualized token activation maps. If empty, visualizations are not saved.
        reso (int): Optional resolution to resize input images; ignored if <= 0 or input is video.

    Returns:
        list: A list of evaluation metric dictionaries, one per input sample.
    """

    # load model
    from qwen_utils import process_vision_info
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    results = []
    for sample_id, (img, prompt, caption, mask, category) in enumerate(tqdm(input_data, unit='sample')):
        # QWen preprocess
        if isinstance(img, list):
            messages = [{"role": "user", "content": [{"type": "video", "video": img}, {"type": "text", "text": prompt}]}]
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if reso > 0 and not isinstance(img, list):
            image_inputs[0] = resize(image_inputs[0], reso)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        # compute logists (CAM) via feat (last hidden state) @ class weights (lm_head). Note: output_logits=True don't return vision logits.
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
        
        generated_ids = outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        # set path and inputs
        if vis_path != '':
            if isinstance(img, list):
                save_dir = os.path.join(vis_path, str(sample_id) + '_' + img[0].split('/')[-2])
            else:
                save_dir = os.path.join(vis_path, str(sample_id) + '_' + img.split('/')[-1].split('.')[0])
            os.makedirs(save_dir, exist_ok=True)

        vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]), 2)]] if isinstance(img, list) else image_inputs

        # set special ids to locate vision tokens, prompt tokens and answer tokens. [start_id (int/list), end_id (int/list)]
        special_ids={'img_id': [151652, 151653], 'prompt_id': [151653, [151645, 198, 151644, 77091]], 'answer_id': [[198, 151644, 77091, 198], -1]}

        # get shape of vision output
        if isinstance(img, list):
            inputs['token_size'] = (inputs['video_grid_thw'][0, 0], inputs['video_grid_thw'][0, 1] // 2, inputs['video_grid_thw'][0, 2] // 2)
        else:
            inputs['token_size'] = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)

        # explain MLLM progressively, draw token activation maps for each round (i is the round idx).
        img_maps, raw_vis_records = [], []
        for i in range(len(logits)):
            # apply TAM to generate the maps and save them if vis_path != "".
            img_map = TAM(generated_ids[0].cpu().tolist(), inputs['token_size'], logits, special_ids, vis_inputs, processor, \
                          os.path.join(save_dir, str(i) + '.jpg') if vis_path != '' else '', i, raw_vis_records, False)
            img_maps.append(img_map)

        # quantitative evaluation
        metrics = evaluate(img_maps, generated_ids_trimmed, processor, caption, mask, category)
        results.append(metrics)

    return results


def eval_llava(model_name='llava-hf/llava-1.5-7b-hf', input_data=[], vis_path=''):
    """
    Evaluate the visual exaplainability of LLaVA and visualize each token and example.

    Args:
        model_name (str): Pretrained model identifier or path.
        input_data (list): List of tuples containing (image, prompt text, caption, mask, category).
        vis_path (str): Directory path to save visualized token activation maps. If empty, visualizations are not saved.

    Returns:
        list: A list of evaluation metric dictionaries, one per input sample.
    """

    # load model
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    results = []
    for sample_id, (img, prompt, caption, mask, category) in enumerate(tqdm(input_data, unit='sample')):
        # preprocess
        image = Image.open(img)
        prompt = "USER: <image>\n%s ASSISTANT:" % (prompt)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

        # inference and compute logits
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, output_hidden_states=True, return_dict_in_generate=True)
        logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

        generated_ids = outputs.sequences
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        # set path and special ids to slice tokens
        if vis_path != '':
            save_dir = os.path.join(vis_path, str(sample_id) + '_' + img.split('/')[-1].split('.')[0])
            os.makedirs(save_dir, exist_ok=True)
        special_ids={'img_id': [[29901, 29871], [29871, 13]], 'prompt_id': [[29871, 13], [319,  1799]], 'answer_id': [[319,  1799,  9047, 13566, 29901], -1]}
        inputs['token_size'] = (24, 24)
       
        # explain MLLM progressively, draw token activation maps for each round (i is the round idx).
        img_maps, raw_vis_records = [], []
        for i in range(len(logits)):
            # apply TAM to generate the maps and save them if vis_path != "".
            img_map = TAM(generated_ids[0].cpu().tolist(), inputs['token_size'], logits, special_ids, image, processor, \
                          os.path.join(save_dir, str(i) + '.jpg') if vis_path != '' else '', i, raw_vis_records, False)
            img_maps.append(img_map)

        # quantitative evaluation
        metrics = evaluate(img_maps, generated_ids_trimmed, processor, caption, mask, category)
        results.append(metrics)

    return results


def prepare_input(dataset_path, processed_input=''):
    """
    Prepare input data for image captioning or description tasks based on the dataset type.

    This function loads and formats input data differently depending on the dataset specified by 
    `dataset_path`. It supports preprocessed input loading, COCO dataset, GranDf/OpenPSG datasets, 
    and datasets organized by attributes.

    Args:
        dataset_path (str): Path to the dataset directory.
        processed_input (str, optional): Filename of a preprocessed input JSON file within the dataset_path.
            If provided and non-empty, the function will load and return this JSON directly.

    Returns:
        list: A list of inputs prepared for the model, where each element is a list containing:
            - Image file path (str)
            - Prompt string (str)
            - Captions or descriptions (list of str)
            - Segmentation label path or empty string (str)
            - Category dictionary or other metadata (dict or other)

    Behavior by dataset type:
    - If `processed_input` is given, loads and returns the JSON content from that file.
    - If `dataset_path` contains "coco":
        Loads COCO minival2014 segmentation and caption annotations.
        For each image, returns the image path, a default caption prompt, a list of captions,
        the segmentation label path, and a category dictionary mapping class names to IDs.
    - If `dataset_path` contains "GranDf" or "OpenPSG":
        Loads `anno.json` and prepares inputs with image paths, a contextual prompt 
        (2 sentences for GranDf, 3 for OpenPSG), a list of descriptions, segmentation label paths, and metadata.
    """

    input_data = []
    if processed_input != '':
        return json.load(open(os.path.join(dataset_path, processed_input)))

    elif 'coco' in dataset_path:
        seg_anno = json.load(open(os.path.join(dataset_path, 'annotations/instances_minival2014.json')))
        cap_anno = json.load(open(os.path.join(dataset_path, 'annotations/captions_val2014.json')))
        defualt_prompt = 'Write a one-sentence caption for this image:' 
        category = {'person': 1,'bicycle': 2,'car': 3,'motorcycle': 4,'airplane': 5,'bus': 6,'train': 7,'truck': 8,'boat': 9,'traffic light': 10,'fire hydrant': 11,'stop sign': 13,'parking meter': 14,'bench': 15,'bird': 16,'cat': 17,'dog': 18,'horse': 19,'sheep': 20,'cow': 21,'elephant': 22,'bear': 23,'zebra': 24,'giraffe': 25,'backpack': 27,'umbrella': 28,'handbag': 31,'tie': 32,'suitcase': 33,'frisbee': 34,'skis': 35,'snowboard': 36,'ball': 37,'kite': 38,'baseball bat': 39,'baseball glove': 40,'skateboard': 41,'surfboard': 42,'tennis racket': 43,'bottle': 44,'glass': 46,'cup': 47,'fork': 48,'knife': 49,'spoon': 50,'bowl': 51,'banana': 52,'apple': 53,'sandwich': 54,'orange': 55,'broccoli': 56,'carrot': 57,'hot dog': 58,'pizza': 59,'donut': 60,'cake': 61,'chair': 62,'couch': 63,'potted plant': 64,'bed': 65,'dining table': 67,'toilet': 70,'tv': 72,'laptop': 73,'mouse': 74,'remote': 75,'keyboard': 76,'cell phone': 77,'microwave': 78,'oven': 79,'toaster': 80,'sink': 81,'refrigerator': 82,'book': 84,'clock': 85,'vase': 86,'scissors': 87,'teddy bear': 88,'hair drier': 89,'toothbrush': 90}
        cap_dic = {}
        for _ in cap_anno['annotations']:
            if _['image_id'] not in cap_dic:
                cap_dic[_['image_id']] = [_['caption']]
            else:
                cap_dic[_['image_id']].append(_['caption'])

        for _ in seg_anno['images']:
            fn = str(_['id']).zfill(12)
            input_data.append([os.path.join(dataset_path, 'image', fn + '.jpg'), defualt_prompt, \
                cap_dic[_['id']], os.path.join(dataset_path, 'seg_label', fn + '.png'), category])

    elif 'GranDf' in dataset_path or 'OpenPSG' in dataset_path:
        data = json.load(open(os.path.join(dataset_path, 'anno.json')))
        if 'GranDf' in dataset_path:
            defualt_prompt = 'Write a description for this image using around two sentences:'
        elif 'OpenPSG' in dataset_path:
            defualt_prompt = 'Write a description for this image using around three sentences:'
        for _ in data:
            input_data.append([os.path.join(dataset_path, _[0]), defualt_prompt, \
                [_[1]], os.path.join(dataset_path, _[2]), _[3]])


    return input_data


def main():
    """
    Main entry point for evaluating multimodal models on a dataset.

    Parses command-line arguments for model name, dataset path, and optional visualization save path.
    Loads and prepares input data, selects the appropriate evaluation function based on the model,
    runs evaluation, aggregates metrics across all samples, and prints summary results.

    Supported models include 'Qwen', 'llava'. If the model is unrecognized,
    a warning message is printed.

    Metrics printed include:
    - Object-level IoU (Obj-IoU)
    - Functional IoU (Func-IoU)
    - F1 score of IoU (F1-IoU)
    - ROUGE-L and METEOR text similarity scores
    - Pixel-level Precision and Recall

    Usage:
        python eval.py <model_name> <dataset_path> [vis_path]

    Args:
        model_name: Pretrained model identifier or path.
        dataset_path: Path to the dataset to evaluate.
        vis_path: Path to save visalizations (optional args).
                  Suggested to vis 100 cases to avoid too much saved images.

    """

    model_name = sys.argv[1]
    dataset_path = sys.argv[2]

    # if vis_path is given, saving TAM images
    try:
        vis_path = sys.argv[3]
    except:
        vis_path = ''

    input_data = prepare_input(dataset_path)
    #input_data = input_data[:100] # vis partial examples
    
    # eval different models (their inference and token ids are different)
    if 'Qwen' in model_name:
        results = eval_qwen2vl(model_name, input_data, vis_path)
    elif 'llava' in model_name:
        results = eval_llava(model_name, input_data, vis_path)
    elif 'InternVL' in model_name:
        print('The models use their codebase, not relased for simplify our code.')
    else:
        print('Lack of custom implementation!')

    # collect the overall results
    res = []
    for i in range(len(results[0])):
        values = []
        for _ in results:
            values.extend(_[i])
        res.append(sum(values) / len(values))

    # print the results. F1-IoU is the harmonic mean value of Obj-IoU and Func-IoU. Pre and Rec are pixel-level metrics besides IoU.
    print('Obj-IoU: %f, Func-IoU: %f, F1-IoU: %f, ROUGE-L: %f, METEOR: %f, Precision: %f, Recall: %f ' \
              % (res[0], res[1], 2* res[0] * res[1] / (res[0] + res[1]), res[2], res[3], res[4], res[5]))


if __name__ == '__main__':
    main()
