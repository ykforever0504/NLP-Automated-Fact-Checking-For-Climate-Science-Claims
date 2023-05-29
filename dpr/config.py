def dpr_setting(args):
    args.epoch = getattr(args, 'epoch', 10)
    args.batch_size = getattr(args, 'batch_size', 8)
    args.report_freqence = getattr(args, "report_freqence", 5)
    args.accumulate_step = getattr(args, "accumulate_step", 4)
    args.model_type = getattr(args, "model_type", "princeton-nlp/sup-simcse-roberta-base")
    # args.model_type = getattr(args, "model_type", "deepset/sentence_bert")
    #about learning rate
    args.max_lr = getattr(args, "max_lr", 2e-5)
    args.lr_steps = getattr(args, "lr_steps", 100)
    args.grad_norm = getattr(args, "grad_norm", 1)
    args.seed = getattr(args, "seed", 999)
    args.max_length = getattr(args, "max_length", 128)
    args.val_interval = getattr(args, "val_interval", 25)
    args.evidence_retrieval_num = getattr(args, "evidence_retrieval_num", 4)
    # a batch all samples
    args.evidence_samples_num = getattr(args, "evidence_samples_num", 64)
