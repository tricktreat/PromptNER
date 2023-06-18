import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
import os
from typing import List
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPProcessor

from prompt4ner import util
from prompt4ner.entities import Dataset, EntityType, RelationType, Entity, Relation, Document, DistributedIterableDataset
from prompt4ner.prompt_tokens import build_prompt_tokens
import copy


class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, processor: CLIPProcessor = None, logger: Logger = None, random_mask_word = None, repeat_gt_entities = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict) 

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i+1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i+1] = entity_type

        # relations
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type
            
        self._datasets = dict()

        self._tokenizer = tokenizer
        self._processor = processor
        self._logger = logger
        self._random_mask_word = random_mask_word
        self._repeat_gt_entities = repeat_gt_entities

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label):
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    def _calc_context_size(self, datasets):
        sizes = [-1]

        for dataset in datasets:
            if isinstance(dataset, Dataset):
                for doc in dataset.documents:
                    sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, processor: CLIPProcessor = None, logger: Logger = None, random_mask_word = False, repeat_gt_entities = None, prompt_length = 3, prompt_type = "soft", prompt_number = 30):
        super().__init__(types_path, tokenizer, processor, logger, random_mask_word, repeat_gt_entities)
        self.prompt_length = prompt_length
        self.prompt_type = prompt_type
        self.prompt_number = prompt_number
        if prompt_type == "hard":
            assert prompt_length == 3
        self.prompt_tokens = build_prompt_tokens(tokenizer)[:prompt_number*prompt_length]
        self.prompt_token_ids = tokenizer.convert_tokens_to_ids(self.prompt_tokens)

        
    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            if dataset_path.endswith(".jsonl"):
                dataset = DistributedIterableDataset(dataset_label, dataset_path, self._relation_types, self._entity_types, self, random_mask_word = self._random_mask_word, tokenizer = self._tokenizer, processor = self._processor, repeat_gt_entities = self._repeat_gt_entities)
                self._datasets[dataset_label] = dataset
            else:
                dataset = Dataset(dataset_label, dataset_path, self._relation_types, self._entity_types, random_mask_word = self._random_mask_word, tokenizer = self._tokenizer, processor = self._processor, repeat_gt_entities = self._repeat_gt_entities)
                self._parse_dataset(dataset_path, dataset, dataset_label)
                self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset, dataset_label):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset: Dataset) -> Document:
        jimages = None
        ltokens = None
        rtokens = None
        jrelations = None

        jtokens = doc['tokens']
        # jrelations = doc['relations']
        jentities = doc['entities']
        if "orig_id" in doc:
            orig_id = doc['orig_id']
        else:
            orig_id = doc['org_id']
        if "ltokens" in doc:
            ltokens = doc["ltokens"]
        if "rtokens" in doc:
            rtokens = doc["rtokens"]
        if "images" in doc and self._processor is not None:
            jimages = doc["images"]

        prompt_number = self.prompt_number
        prompt_tokens = self.prompt_tokens
        # context_word = "is"
        if self.prompt_length <= 2:
            common_token = 0
        else:
            common_token = self.prompt_length-2

        if self.prompt_length>0:
            prompt = ["{}"] * self.prompt_length
            if self.prompt_type == "hard":
                prompt[1] = "is"
                common_token = 0
            else:
                for i in range(common_token):
                    prompt[i+1] = f"{prompt_tokens[-i]}"
            prompt = " ".join(prompt*prompt_number).format(*[prompt_tokens[i] for i in range(0, prompt_number*self.prompt_length)])
            prompt = prompt.split(" ")
        else:
            prompt = []

        images = []
        if jimages is not None:
            images = self._parse_images(jimages, orig_id, dataset)

        # parse tokens
        doc_tokens, doc_encoding, seg_encoding, raw_doc_encoding, pos_encoding, inx4locator = self._parse_tokens(jtokens, ltokens, rtokens, prompt, dataset)

        if len(doc_encoding) > 512:
            self._log(f"Document {doc['orig_id']} len(doc_encoding) = {len(doc_encoding) } > 512, Ignored!")
            return None
        
        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # parse relations
        relations = []
        if jrelations is not None:
            relations = self._parse_relations(jrelations, entities, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding, seg_encoding, raw_doc_encoding, inx4locator, pos_encoding, images = images)

        return document

    def _parse_images(self, jimages, org_id, dataset: Dataset, num = 1):
        images = []
        local_dir = "/".join(dataset._path.split("/")[:-1])+"/images/"

        # if not is_cached:
        for jimage in jimages:
            path = local_dir+str(jimage['id'])+".jpg"
            if os.path.exists(path):
                image = dataset.create_image(jimage['url'], jimage['caption'], jimage['id'], jimage['similarity'], local_dir)
                images.append(image)

            if len(images)>=num:
                break
            
        while len(images)<num:
            image = dataset.create_image("", "", 0, 0, local_dir)
            images.append(image)

            # with open(cache_path, 'wb') as f:
            #     torch.save(images, f)
        assert len(images) == num
        return images

    def _parse_tokens(self, jtokens, ltokens, rtokens, prompt, dataset):
        doc_tokens = []

        special_tokens_map = self._tokenizer.special_tokens_map
        doc_encoding = [self._tokenizer.convert_tokens_to_ids(special_tokens_map['cls_token'])]
        seg_encoding = [1]

        if ltokens is not None and len(ltokens)>0:
            for token_phrase in ltokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]
        
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)

            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding) - 1 )
            span_start = span_start + len(prompt)
            span_end = span_end + len(prompt)
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding
            seg_encoding += [1] * len(token_encoding)
        
        if rtokens is not None and len(rtokens)>0:
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]
            for token_phrase in rtokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)

        doc_encoding = doc_encoding[:512-self.prompt_number*self.prompt_length]
        seg_encoding = seg_encoding[:512-self.prompt_number*self.prompt_length]

        raw_doc_encoding = copy.deepcopy(doc_encoding)
        
        for token_phrase in prompt:
            token_encoding = self._tokenizer.convert_tokens_to_ids(token_phrase)
            doc_encoding.insert(0, token_encoding)
            seg_encoding.insert(0, 0)

        pos_encoding = list(range(self.prompt_number*self.prompt_length)) + list(range(len(raw_doc_encoding)))
        
        inx4locator = None
        if self.prompt_length>0:
            inx4locator = np.array(range(0, self.prompt_number*self.prompt_length, self.prompt_length))

        return doc_tokens, doc_encoding, seg_encoding, raw_doc_encoding, pos_encoding, inx4locator

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention  (exclusive)
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            head = entities[head_idx]
            tail = entities[tail_idx]

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations
