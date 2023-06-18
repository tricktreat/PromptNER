import copy
import torch
from torch import nn as nn
from torch.nn import functional as F
from prompt4ner.modeling_albert import AlbertModel, AlbertMLMHead
from prompt4ner.modeling_bert import BertConfig, BertModel
from prompt4ner.modeling_roberta import RobertaConfig, RobertaModel, RobertaLMHead
from prompt4ner.modeling_xlm_roberta import XLMRobertaConfig
from transformers.modeling_utils import PreTrainedModel
from prompt4ner import util
import logging

logger = logging.getLogger()

class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.tanh(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -1e25
        # entity_token_p = entity_token_cls.softmax(dim=-1)
        entity_token_p = F.sigmoid(entity_token_cls)
        return entity_token_p

class EntityBoundaryPredictorBak(nn.Module):
    def __init__(self, config, prop_drop):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(prop_drop)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(prop_drop)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -1e30
        entity_token_p = F.sigmoid(entity_token_cls)

        return entity_token_p


class EntityTypePredictor(nn.Module):
    def __init__(self, config, entity_type_count, mlm_head):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, entity_type_count),
        )
    
    def forward(self, h_cls):
        entity_logits = self.classifier(h_cls)
        return entity_logits

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, selfattn = True, ffn = True):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.selfattn = selfattn
        self.ffn = ffn

        if selfattn:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
        if ffn:
            # ffn
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.dropout3 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout4 = nn.Dropout(dropout)
            self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, pos, src, mask):
        if self.selfattn:
            q = k = self.with_pos_embed(tgt, pos)
            v = tgt
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        q = self.with_pos_embed(tgt, pos)
        k = v = src
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask if mask is not None else None)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.ffn:
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm3(tgt)

        return tgt

class DetrTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, pos, src, mask):
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            output = layer(output, tgt, src, mask)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output

class Prompt4NER(PreTrainedModel):

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _compute_extended_attention_mask(self, attention_mask, context_count, prompt_number):
        
        if not self.prompt_individual_attention and not self.sentence_individual_attention:
            # #batch x seq_len
            extended_attention_mask = attention_mask
        else:
            # #batch x seq_len x seq_len
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(-1), -1).clone()

            for mask, c_count in zip(extended_attention_mask, context_count):
                # mask seq_len x seq_len
                # mask prompt for sentence encoding
                if self.prompt_individual_attention:
                    # encode for each prompt
                    for p in range(prompt_number):
                        mask[p*self.prompt_length:  p*self.prompt_length + self.prompt_length, :prompt_number*self.prompt_length] = 0
                        mask[p*self.prompt_length: p*self.prompt_length + self.prompt_length, p*self.prompt_length: p*self.prompt_length + self.prompt_length] = 1
                if self.sentence_individual_attention:
                    for c in range(c_count):
                        mask[c+self.prompt_length*prompt_number, :self.prompt_length*prompt_number] = 0

        return extended_attention_mask

    def __init__(
        self,
        model_type, 
        config, 
        entity_type_count: int, 
        prop_drop: float, 
        freeze_transformer: bool, 
        lstm_layers = 3, 
        decoder_layers = 3,
        pool_type:str = "max", 
        prompt_individual_attention = True, 
        sentence_individual_attention = True,
        use_masked_lm = False, 
        last_layer_for_loss = 3, 
        split_epoch = 0, 
        clip_v = None,
        prompt_length = 3,
        prompt_number = 60,
        prompt_token_ids = None):
        super().__init__(config)
        
        self.freeze_transformer = freeze_transformer
        self.split_epoch = split_epoch
        self.has_changed = False
        self.loss_layers = last_layer_for_loss
        self.model_type = model_type
        self.use_masked_lm = use_masked_lm
        self._entity_type_count = entity_type_count
        self.prop_drop = prop_drop
        self.split_epoch = split_epoch
        self.lstm_layers = lstm_layers
        self.prompt_individual_attention = prompt_individual_attention
        self.sentence_individual_attention = sentence_individual_attention

        self.decoder_layers = decoder_layers
        self.prompt_number = prompt_number
        self.prompt_length = prompt_length
        self.pool_type = pool_type

        self.withimage = False
        if clip_v is not None:
            self.withimage = True
        
        self.query_embed = nn.Embedding(prompt_number, config.hidden_size * 2)

        if self.decoder_layers > 0:
            decoder_layer = DetrTransformerDecoderLayer(d_model=config.hidden_size, d_ffn=1024, dropout=0.1, selfattn = True, ffn = True)
            self.decoder = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)
            if self.withimage:
                self.img_decoder = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)
            if self.prompt_length>1:
                self.decoder2 = DetrTransformerDecoder(decoder_layer=decoder_layer, num_layers=self.decoder_layers)

        if model_type == "roberta":
            self.roberta = RobertaModel(config)
            self.model = self.roberta
            self.lm_head = RobertaLMHead(config)
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.lm_head(x))

        if model_type == "bert":
            # self.bert = BertModel(config)
            self.prompt_ids = prompt_token_ids
            self.bert = BertModel(config, prompt_ids = prompt_token_ids)
            self.model = self.bert
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = False
            # self.cls = BertOnlyMLMHead(config)
            self.cls = None
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.cls(x))

        if model_type == "albert":
            # self.bert = BertModel(config)
            self.prompt_ids = prompt_token_ids
            self.bert = AlbertModel(config)
            self.model = self.bert
            self.predictions = AlbertMLMHead(config)
            self.entity_classifier = EntityTypePredictor(config, entity_type_count, lambda x: self.predictions(x))

        if self.withimage:
            self.vision2text = nn.Linear(clip_v.config.hidden_size, config.hidden_size)

        Prompt4NER._keys_to_ignore_on_save = ["model." + k for k,v in self.model.named_parameters()]
        # Prompt4NER._keys_to_ignore_on_load_unexpected = ["model." + k for k,v in self.model.named_parameters()]
        Prompt4NER._keys_to_ignore_on_load_missing = ["model." + k for k,v in self.model.named_parameters()]

        if self.lstm_layers > 0:
            self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size//2, num_layers = lstm_layers,  bidirectional = True, dropout = 0.1, batch_first = True)

        self.left_boundary_classfier = EntityBoundaryPredictor(config, self.prop_drop)
        self.right_boundary_classfier = EntityBoundaryPredictor(config, self.prop_drop)
        # self.entity_classifier = EntityTypePredictor(config, config.hidden_size, entity_type_count)
        self.init_weights()

        self.clip_v = clip_v

        if freeze_transformer or self.split_epoch > 0:
            logger.info("Freeze transformer weights")
            if self.model_type == "bert":
                model = self.bert
                mlm_head = self.cls
            if self.model_type == "roberta":
                model = self.roberta
                mlm_head = self.lm_head
            if self.model_type == "albert":
                model = self.albert
                mlm_head = self.predictions
            for name, param in model.named_parameters():
                param.requires_grad = False

    def _common_forward(
        self, 
        encodings: torch.tensor, 
        context_masks: torch.tensor, 
        raw_context_masks: torch.tensor, 
        inx4locator: torch.tensor, 
        pos_encoding: torch.tensor, 
        seg_encoding: torch.tensor, 
        context2token_masks:torch.tensor,
        token_masks:torch.tensor,
        image_inputs: dict = None,
        meta_doc = None):
        
        batch_size = encodings.shape[0]
        context_masks = context_masks.float()
        token_count = token_masks.long().sum(-1,keepdim=True)
        context_count = context_masks.long().sum(-1,keepdim=True)
        raw_context_count = raw_context_masks.long().sum(-1,keepdim=True)
        pos = None
        tgt = None
        tgt2 = None

        # pdb.set_trace()
        
        context_masks = self._compute_extended_attention_mask(context_masks, raw_context_count, self.prompt_number)
        # self = self.eval()
        if self.model_type == "bert":
            model = self.bert
        if self.model_type == "roberta":
            model = self.roberta
        # model.embeddings.position_embeddings
        outputs = model(
                    input_ids=encodings,
                    attention_mask=context_masks,
                    # token_type_ids=seg_encoding,
                    # position_ids=pos_encoding,
                    output_hidden_states=True)
        # last_hidden_state, pooler_output, hidden_states 

        masked_seq_logits = None
        if self.use_masked_lm and self.training:
            if self.model_type == "bert":
                masked_seq_logits = self.cls(outputs.last_hidden_state)
            if self.model_type == "roberta":
                masked_seq_logits = self.lm_head(outputs.last_hidden_state)

        if self.withimage:
            image_h = self.clip_v(**image_inputs)
            image_last_hidden_state = image_h.last_hidden_state
            aligned_image_h = self.vision2text(image_last_hidden_state)
        
        query_embed = self.query_embed.weight # [100, 768 * 2]
        query_embeds = torch.split(query_embed, outputs.last_hidden_state.size(2), dim=-1)
        

        if tgt is None:
            tgt = query_embeds[1]
            tgt = tgt.unsqueeze(0).expand(batch_size, -1, -1) # [2, 100, 768]

        if pos is None:
            pos = query_embeds[0]
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1) # [2, 100, 768]

        orig_tgt = tgt
        intermediate = []
        for i in range(self.loss_layers, 0, -1):

            h = outputs.hidden_states[-1]
            h_token = util.combine(h, context2token_masks, self.pool_type)

            if self.lstm_layers > 0:
                h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
                h_token, (_, _) = self.lstm(h_token)
                h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)

            if inx4locator is not None:
                
                tgt = util.batch_index(outputs.hidden_states[-i], inx4locator) + orig_tgt
                if self.prompt_length > 1:
                    tgt2 = util.batch_index(outputs.hidden_states[-i], inx4locator + self.prompt_length-1) + orig_tgt

            updated_tgt = tgt

            if tgt2 is None:
                updated_tgt2 = tgt
            else:
                updated_tgt2 = tgt2

            if self.decoder_layers > 0:
                if self.withimage:
                    tgt = self.img_decoder(tgt, pos, aligned_image_h, mask = None)
                updated_tgt = self.decoder(tgt, pos, h_token, mask = token_masks)

                if self.prompt_length > 1:
                    updated_tgt2 = self.decoder2(tgt2, pos, h_token, mask = token_masks)
                else:
                    updated_tgt2 = updated_tgt
            intermediate.append({"h_token":h_token, "left_h_locator":updated_tgt, "right_h_locator":updated_tgt, "h_cls":updated_tgt2})

        output = []
        
        for h_dict in intermediate:
            h_token, left_h_locator, right_h_locator, h_cls = h_dict["h_token"], h_dict["left_h_locator"], h_dict["right_h_locator"], h_dict["h_cls"]
            p_left = self.left_boundary_classfier(h_token, left_h_locator, token_masks)
            p_right = self.right_boundary_classfier(h_token, right_h_locator, token_masks)
            entity_logits = self.entity_classifier(h_cls)
            output.append({"p_left": p_left, "p_right": p_right, "entity_logits": entity_logits})

        return entity_logits, p_left, p_right, masked_seq_logits, output
    
    def _forward_train(self, *args, epoch=0, **kwargs):
        if not self.has_changed and epoch >= self.split_epoch and not self.freeze_transformer:
            logger.info(f"Now, update bert weights @ epoch = {self.split_epoch }")
            self.has_changed = True
            for name, param in self.named_parameters():
                param.requires_grad = True

        return self._common_forward(*args, **kwargs)

    def _forward_eval(self, *args, **kwargs):
        return self._common_forward(*args, **kwargs)

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

class BertPrompt4NER(Prompt4NER):
    
    config_class = BertConfig
    base_model_prefix = "bert"
    # base_model_prefix = "model"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("bert", *args, **kwagrs)

class RobertaPrompt4NER(Prompt4NER):

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)
    
class XLMRobertaPrompt4NER(Prompt4NER):

    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)

class AlbertPrompt4NER(Prompt4NER):

    config_class = XLMRobertaConfig
    base_model_prefix = "albert"
    # base_model_prefix = "model"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("albert", *args, **kwagrs)


_MODELS = {
    'prompt4ner': BertPrompt4NER,
    'roberta_prompt4ner': RobertaPrompt4NER,
    'xlmroberta_prompt4ner': XLMRobertaPrompt4NER,
    'albert_prompt4ner': AlbertPrompt4NER
}

def get_model(name):
    return _MODELS[name]
