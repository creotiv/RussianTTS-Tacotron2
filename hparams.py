from tools import HParams
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=3000,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        mmi_ignore_layers=["decoder.linear_projection.linear_layer.weight", "decoder.linear_projection.linear_layer.bias", "decoder.gate_layer.linear_layer.weight"],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        dataset_path='/home/creotiv/work/tts/Natasha/',
        training_files='/home/creotiv/work/tts/Natasha/filelists/train_style.csv',
        validation_files='/home/creotiv/work/tts/Natasha/filelists/val.csv',
        text_cleaners=['transliteration_cleaners_with_stress'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1500,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_teacher_forcing=1.0,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # GST ======================
        # Reference encoder
        use_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer
        token_embedding_size=256,
        token_num=10,
        num_heads=8,

        no_dga=False,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=16,
        mask_padding=True,  # set model's padded outputs to padded values

        ################################
        # FINE-TUNE #
        ################################
        use_mmi=False,

        drop_frame_rate=0.0 ,#0.2,
        use_gaf=False,
        update_gaf_every_n_step=10,
        max_gaf=0.5,
        
        global_mean_npy='ruslan_global_mean.npy',
        
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
