from transformers.configuration_utils import PretrainedConfig


class MeterConfig(PretrainedConfig):
    model_type = "meter"

    def __init__(
        self,
        # Text Setting
        vqa_label_size=3129,
        max_text_len=40,
        tokenizer="roberta-base",
        vocab_size=50265,
        whole_word_masking=False,
        mlm_prob=0.15,
        draw_false_text=0,
        token_types=2,
        prune_layers=None,
        keep_ratio=1,
        merge_ratio=0,
        merge_r=0,
        prune_r=0,
        sim_method="mean_head",
        merge_style="tip",
        merge_text=0,
        reduce_layers=None,
        contrast_method=None,
        ib_kl=False,
        # Transformer Setting
        image_size=288,
        vit_patch_size=16,
        num_top_layer=6,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4,
        drop_rate=0.1,
        initializer_range=0.02,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        # Text Setting
        self.vqa_label_size = vqa_label_size
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.whole_word_masking = whole_word_masking
        self.mlm_prob = mlm_prob
        self.draw_false_text = draw_false_text
        self.token_types = token_types
        self.prune_layers = prune_layers
        self.contrast_method = contrast_method
        self.ib_kl = ib_kl

        # Transformer Setting
        self.image_size = image_size
        self.vit_patch_size = vit_patch_size
        self.num_top_layer = num_top_layer
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate

        self.keep_ratio = keep_ratio
        self.sim_method = sim_method
        self.merge_ratio = merge_ratio
        self.merge_style = merge_style
        self.merge_r = merge_r
        self.prune_r = prune_r
        self.merge_text = merge_text
        self.reduce_layers = reduce_layers


class MeterForPreTrainingConfig(MeterConfig):
    def __init__(self, itm_loss=1, mlm_loss=1, mpp_loss=0, **kwargs):
        super().__init__(**kwargs)
        self.itm_loss = itm_loss
        self.mlm_loss = mlm_loss
        self.mpp_loss = mpp_loss
