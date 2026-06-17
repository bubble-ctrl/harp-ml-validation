"""
manual_test_dataset.py — 40 meticulously crafted real-world test cases
for GNC and Merge API validation.

Covers:
  GNC: Genuine (ResNet, LSTM, Clipping), FP (fixed, accum, GAN, no optim, nested),
       Bad refactors (outside loop, after backward).
  Merge: Genuine (left/outer required, validate missing), FP (explicit, index merge),
         Bad refactors (hallucinated key, wrong how, missing suffixes).

Author: Your Name
Date: 2026-06-17
"""

TEST_CASES = {

    # =====================================================================
    #  GNC (Gradients Not Cleared) — 20 cases
    # =====================================================================

    "GNC_001": {
        "description": "Genuine smell: ResNet-50 training on ImageNet-style loader. Missing zero_grad.",
        "reasoning": "Stage1 should return GENUINE. Correct fix adds zero_grad at loop top. "
                     "C4 (grad identity) will fail against original, but AST override accepts.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_resnet(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": """
def train_resnet(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": """
def train_resnet(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    optimizer.zero_grad()   # WRONG: placed OUTSIDE the inner loop
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
"""
    },

    "GNC_002": {
        "description": "Genuine smell: LSTM for language modeling with packed sequences. Missing zero_grad.",
        "reasoning": "The context includes pack_padded_sequence — the AST must traverse nested calls. "
                     "Stage1 should still flag GENUINE.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_lstm_lm(epochs=3):
    model = nn.LSTM(vocab_size, 512, num_layers=2, batch_first=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loader = [(torch.randint(0,1000,(4,50)), torch.randint(0,1000,(4,50))) for _ in range(10)]
    for epoch in range(epochs):
        for src, tgt in loader:
            src_len = (src != pad_idx).sum(dim=1)
            packed_src = pack_padded_sequence(src, src_len.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = model(packed_src)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            loss = criterion(out.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": """
def train_lstm_lm(epochs=3):
    model = nn.LSTM(vocab_size, 512, num_layers=2, batch_first=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loader = [(torch.randint(0,1000,(4,50)), torch.randint(0,1000,(4,50))) for _ in range(10)]
    for epoch in range(epochs):
        for src, tgt in loader:
            optimizer.zero_grad()
            src_len = (src != pad_idx).sum(dim=1)
            packed_src = pack_padded_sequence(src, src_len.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = model(packed_src)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            loss = criterion(out.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": """
def train_lstm_lm(epochs=3):
    model = nn.LSTM(vocab_size, 512, num_layers=2, batch_first=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    loader = [(torch.randint(0,1000,(4,50)), torch.randint(0,1000,(4,50))) for _ in range(10)]
    for epoch in range(epochs):
        for src, tgt in loader:
            src_len = (src != pad_idx).sum(dim=1)
            packed_src = pack_padded_sequence(src, src_len.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = model(packed_src)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            loss = criterion(out.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   # WRONG: zero_grad after step — still accumulates within batch
    return model
"""
    },

    "GNC_003": {
        "description": "Genuine smell: Training with gradient clipping. Missing zero_grad.",
        "reasoning": "Clipping is present, but zero_grad is still required. AST sees clip_grad_norm_ "
                     "but should not treat it as a false positive.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_with_clip(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,128), torch.randn(4,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
""",
        "after_correct": """
def train_with_clip(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,128), torch.randn(4,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
""",
        "after_wrong": """
def train_with_clip(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,128), torch.randn(4,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.zero_grad()   # WRONG: zero_grad after backward — erases the gradients before step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model
"""
    },

    "GNC_004": {
        "description": "False Positive (Cat 1): zero_grad already correctly placed at the top of the inner loop.",
        "reasoning": "AST finds zero_grad in the same scope as backward. Stage1 returns NO_SMELL.",
        "smell_type": "gnc",
        "expected_stage1": "NO_SMELL",
        "expected_validator": "NO_SMELL",
        "before": """
def train_already_good(epochs=2):
    model = nn.Linear(64, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,64), torch.randn(4,1)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_005": {
        "description": "False Positive (Cat 2): Intentional gradient accumulation with accumulation_steps variable.",
        "reasoning": "Stage1 sees 'accumulation_steps' variable and modulo gate. Returns AMBIGUOUS with high confidence.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def train_accum(epochs=2, accumulation_steps=4):
    model = nn.Linear(32, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,32), torch.randn(4,1)) for _ in range(20)]
    for epoch in range(epochs):
        optimizer.zero_grad()
        for step, (X, y) in enumerate(loader):
            loss = criterion(model(X), y) / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    return model
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_006": {
        "description": "False Positive (Cat 2): Modulo gate with variable named 'grad_accum'.",
        "reasoning": "Alternative naming of accumulation variable. AST uses keyword set to catch this.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def train_bert_accum(epochs=1, grad_accum=8):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loader = [(torch.randint(0,100,(4,128)), torch.randint(0,2,(4,))) for _ in range(20)]
    optimizer.zero_grad()
    for step, (input_ids, labels) in enumerate(loader):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
    return model
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_007": {
        "description": "False Positive (Cat 3): Multi-optimizer GAN training.",
        "reasoning": "Multiple optimizer objects (optimizer_G, optimizer_D) with separate zero_grad calls. Stage1 returns AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def train_gan(epochs=2):
    generator = nn.Linear(100, 784)
    discriminator = nn.Linear(784, 1)
    opt_G = optim.Adam(generator.parameters(), lr=0.0002)
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCEWithLogitsLoss()
    loader = [(torch.randn(4,784),) for _ in range(10)]
    for epoch in range(epochs):
        for real_imgs in loader:
            # Train Discriminator
            opt_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), torch.ones(4,1))
            fake = generator(torch.randn(4,100))
            fake_loss = criterion(discriminator(fake.detach()), torch.zeros(4,1))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            opt_D.step()
            # Train Generator
            opt_G.zero_grad()
            g_loss = criterion(discriminator(fake), torch.ones(4,1))
            g_loss.backward()
            opt_G.step()
    return generator, discriminator
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_008": {
        "description": "False Positive (Cat 4): No optimizer present — computing adversarial perturbation (DeepFool style).",
        "reasoning": "loss.backward() called solely to compute input gradients. No optimizer or zero_grad anywhere. Stage1 returns AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def deepfool_attack(model, image, label, max_iter=10):
    image = image.clone().detach().requires_grad_(True)
    for _ in range(max_iter):
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward(retain_graph=True)
        grad = image.grad.clone()
        # ... perturbation logic ...
        image.grad.zero_()
    return image
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_009": {
        "description": "False Positive (Cat 4): Gradient saliency maps for interpretability (no optimizer).",
        "reasoning": "Used for Grad-CAM / integrated gradients. No training loop. Stage1 returns AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def compute_saliency(model, input_tensor, target_class):
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    loss = output[0, target_class]
    loss.backward()
    saliency = input_tensor.grad.abs().squeeze()
    return saliency
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_010": {
        "description": "Ambiguous: Nested backward with custom gradient hooks (meta-learning / MAML style).",
        "reasoning": "backward() is called inside an inner loop deeper than the optimizer.step(). Stage1 sees nested_depth > step_depth -> AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def maml_inner_loop(model, support_set, query_set, inner_lr=0.01):
    fast_weights = {k: p.clone() for k, p in model.named_parameters()}
    for x, y in support_set:
        out = model.functional_forward(x, fast_weights)
        loss = nn.MSELoss()(out, y)
        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
        for (k, p), g in zip(fast_weights.items(), grads):
            fast_weights[k] = p - inner_lr * g
    out = model.functional_forward(query_set, fast_weights)
    loss = nn.MSELoss()(out, y_query)
    loss.backward()  # Nested grad computation — not a standard training loop
    return loss
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_011": {
        "description": "Ambiguous: Multiple backwards in a custom loss (e.g., contrastive loss with hooks).",
        "reasoning": "Similar to Figure 8 in the paper — backward called inside a nested hook. AST flags as AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "AMBIGUOUS",
        "expected_validator": "AMBIGUOUS",
        "before": """
def custom_backward_hook(grad_output, cache):
    for reps_mb, grad_mb in zip(cache['embeds'], cache['grads']):
        surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
        surrogate.backward(retain_graph=True)
    return None
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_012": {
        "description": "Bad refactoring: zero_grad placed OUTSIDE the outer training loop.",
        "reasoning": "Stage1 sees zero_grad exists, but AST scope analysis shows it's not in the same loop as backward(). "
                     "Validator should return REGRESSION.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",  # Stage1 only checks before; validation catches the bad refactor
        "expected_validator": "REGRESSION",
        "before": """
def train_bad_scope(epochs=2):
    model = nn.Linear(10,1)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loader = [(torch.randn(4,10), torch.randn(4,1)) for _ in range(5)]
    for epoch in range(epochs):
        for X, y in loader:
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    return model
""",
        "after_correct": "",
        "after_wrong": """
def train_bad_scope(epochs=2):
    model = nn.Linear(10,1)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loader = [(torch.randn(4,10), torch.randn(4,1)) for _ in range(5)]
    opt.zero_grad()   # WRONG: placed OUTSIDE the loop
    for epoch in range(epochs):
        for X, y in loader:
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    return model
"""
    },

    "GNC_013": {
        "description": "Bad refactoring: zero_grad placed AFTER loss.backward() but BEFORE optimizer.step().",
        "reasoning": "This still zeroes out the gradients before the step is taken (no parameter update). "
                     "C1 (execution) passes, but C4 (grad identity) or C3 (loss trend) fails -> REGRESSION.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def train_resnet(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": "",
        "after_wrong": """
def train_bad_order(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.zero_grad()   # WRONG: erases gradients before step
            optimizer.step()
    return model
"""
    },

    "GNC_014": {
        "description": "Bad refactoring: zero_grad placed only inside a conditional that may not execute.",
        "reasoning": "AST shows zero_grad exists, but it's gated — not guaranteed to execute every iteration. "
                     "Validator should mark as REGRESSION or AMBIGUOUS.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def train_resnet(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": """
def train_conditional_zg(epochs=2):
    model = models.resnet50(pretrained=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    train_loader = [(torch.randn(4,3,224,224), torch.randint(0,1000,(4,))) for _ in range(10)]
    model.train()
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:   # Only zeroes every 100 steps — not sufficient
                optimizer.zero_grad()
    return model
"""
    },

    "GNC_015": {
        "description": "Genuine smell: Multi-GPU DataParallel training (loss computed per GPU).",
        "reasoning": "Standard pattern with DataParallel. Missing zero_grad in the loop. Stage1 flags GENUINE.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_dp(epochs=2):
    model = nn.DataParallel(nn.Linear(128, 10))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = [(torch.randn(8,128), torch.randn(8,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": """
def train_dp(epochs=2):
    model = nn.DataParallel(nn.Linear(128, 10))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = [(torch.randn(8,128), torch.randn(8,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": ""
    },

    "GNC_016": {
        "description": "Genuine smell: Training with learning rate scheduler (ReduceLROnPlateau).",
        "reasoning": "Scheduler.step() is called, but zero_grad is missing. Scheduler calls do not affect gradient clearing.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_with_scheduler(epochs=2):
    model = nn.Linear(64, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,64), torch.randn(4,1)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
    return model
""",
        "after_correct": """
def train_with_scheduler(epochs=2):
    model = nn.Linear(64, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,64), torch.randn(4,1)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
    return model
""",
        "after_wrong": ""
    },

    "GNC_017": {
        "description": "Genuine smell: Training with automatic mixed precision (AMP) — missing zero_grad.",
        "reasoning": "Scaler is used, but zero_grad is still required. AST sees scaler but should not treat as FP.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_amp(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,128), torch.randn(4,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model
""",
        "after_correct": """
def train_amp(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,128), torch.randn(4,10)) for _ in range(10)]
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model
""",
        "after_wrong": ""
    },

    "GNC_018": {
        "description": "Genuine smell: Training with multiple losses (auxiliary classifier).",
        "reasoning": "Two losses backward separately, but zero_grad is missing. AST should catch this.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def train_aux_loss(epochs=2):
    model = nn.Sequential(nn.Linear(128,64), nn.Linear(64,10))
    aux_head = nn.Linear(64,5)
    optimizer = optim.Adam(list(model.parameters()) + list(aux_head.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = [(torch.randn(4,128), torch.randint(0,10,(4,)), torch.randint(0,5,(4,))) for _ in range(10)]
    for epoch in range(epochs):
        for X, y_main, y_aux in loader:
            feat = model[0](X)
            main_out = model[1](feat)
            aux_out = aux_head(feat)
            loss = criterion(main_out, y_main) + 0.3 * criterion(aux_out, y_aux)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": """
def train_aux_loss(epochs=2):
    model = nn.Sequential(nn.Linear(128,64), nn.Linear(64,10))
    aux_head = nn.Linear(64,5)
    optimizer = optim.Adam(list(model.parameters()) + list(aux_head.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = [(torch.randn(4,128), torch.randint(0,10,(4,)), torch.randint(0,5,(4,))) for _ in range(10)]
    for epoch in range(epochs):
        for X, y_main, y_aux in loader:
            optimizer.zero_grad()
            feat = model[0](X)
            main_out = model[1](feat)
            aux_out = aux_head(feat)
            loss = criterion(main_out, y_main) + 0.3 * criterion(aux_out, y_aux)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": ""
    },

    "GNC_019": {
        "description": "False Positive (Cat 1): zero_grad present but code contains 'backward' in comments.",
        "reasoning": "AST must differentiate actual code from strings/comments. This test ensures comments don't cause false flags.",
        "smell_type": "gnc",
        "expected_stage1": "NO_SMELL",
        "expected_validator": "NO_SMELL",
        "before": """
def train_with_comment(epochs=2):
    model = nn.Linear(10,1)
    opt = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    loader = [(torch.randn(4,10), torch.randn(4,1)) for _ in range(5)]
    for epoch in range(epochs):
        for X, y in loader:
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()  # This is the backward call
            opt.step()
            # Note: zero_grad is already called above.
    return model
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "GNC_020": {
        "description": "Genuine smell: Training with deep learning library 'pytorch_lightning' style (explicit training_step).",
        "reasoning": "Even in Lightning, manual loops can miss zero_grad. This tests a longer function context.",
        "smell_type": "gnc",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def lightning_style_training(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters())
    train_loader = DataLoader(TensorDataset(torch.randn(100,128), torch.randint(0,10,(100,))), batch_size=4)
    for epoch in range(epochs):
        for batch in train_loader:
            X, y = batch
            logits = model(X)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_correct": """
def lightning_style_training(epochs=2):
    model = nn.Linear(128, 10)
    optimizer = optim.Adam(model.parameters())
    train_loader = DataLoader(TensorDataset(torch.randn(100,128), torch.randint(0,10,(100,))), batch_size=4)
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch
            logits = model(X)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    return model
""",
        "after_wrong": ""
    },

    # =====================================================================
    #  MERGE API — 20 cases
    # =====================================================================

    "MERGE_001": {
        "description": "Genuine smell: Default inner join on customers/orders. Business needs LEFT join to retain all customers.",
        "reasoning": "DIFFERENTIAL PBT FAILS (row counts differ). CONTRACT PBT passes (left_rows_preserved=true). "
                     "This is the silver bullet case proving why contract validation is necessary.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def get_customer_orders():
    customers = pd.DataFrame({'customer_id': [1,2,3,4], 'name': ['A','B','C','D']})
    orders = pd.DataFrame({'cust_id': [1,2,2], 'amount': [100,200,300]})
    # SMELL: default inner join drops customers with no orders (C and D)
    result = customers.merge(orders, left_on='customer_id', right_on='cust_id')
    return result
""",
        "after_correct": """
def get_customer_orders():
    customers = pd.DataFrame({'customer_id': [1,2,3,4], 'name': ['A','B','C','D']})
    orders = pd.DataFrame({'cust_id': [1,2,2], 'amount': [100,200,300]})
    # CORRECT: left join preserves all customers
    result = customers.merge(orders, left_on='customer_id', right_on='cust_id', how='left', validate='one_to_many')
    return result
""",
        "after_wrong": """
def get_customer_orders():
    customers = pd.DataFrame({'customer_id': [1,2,3,4], 'name': ['A','B','C','D']})
    orders = pd.DataFrame({'cust_id': [1,2,2], 'amount': [100,200,300]})
    # WRONG: SLM hallucinates 'id' as the key
    result = customers.merge(orders, left_on='id', right_on='id', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_002": {
        "description": "Genuine smell: Merging sales with products. Missing validate='many_to_one' — potential duplicates.",
        "reasoning": "Contract PBT should infer validate from schema (sales has multiple rows per product).",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def sales_product_merge():
    sales = pd.DataFrame({'product_id': [101,102,101,103], 'qty': [1,2,3,4]})
    products = pd.DataFrame({'sku': [101,102,103], 'price': [10,20,30]})
    result = sales.merge(products, left_on='product_id', right_on='sku')
    return result
""",
        "after_correct": """
def sales_product_merge():
    sales = pd.DataFrame({'product_id': [101,102,101,103], 'qty': [1,2,3,4]})
    products = pd.DataFrame({'sku': [101,102,103], 'price': [10,20,30]})
    result = sales.merge(products, left_on='product_id', right_on='sku', how='inner', validate='many_to_one')
    return result
""",
        "after_wrong": """
def sales_product_merge():
    sales = pd.DataFrame({'product_id': [101,102,101,103], 'qty': [1,2,3,4]})
    products = pd.DataFrame({'sku': [101,102,103], 'price': [10,20,30]})
    # WRONG: SLM guesses validate='one_to_one' — will fail if duplicate matches exist
    result = sales.merge(products, left_on='product_id', right_on='sku', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_003": {
        "description": "Genuine smell: Merging on multiple columns (composite key) with defaults.",
        "reasoning": "Tests that SLM-2 can recover multi-column keys from context.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_composite():
    left = pd.DataFrame({'year': [2020,2020,2021], 'month': [1,2,1], 'val': [10,20,30]})
    right = pd.DataFrame({'yr': [2020,2020,2021], 'mon': [1,2,1], 'metric': [100,200,300]})
    result = left.merge(right)
    return result
""",
        "after_correct": """
def merge_composite():
    left = pd.DataFrame({'year': [2020,2020,2021], 'month': [1,2,1], 'val': [10,20,30]})
    right = pd.DataFrame({'yr': [2020,2020,2021], 'mon': [1,2,1], 'metric': [100,200,300]})
    result = left.merge(right, left_on=['year','month'], right_on=['yr','mon'], how='inner', validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_composite():
    left = pd.DataFrame({'year': [2020,2020,2021], 'month': [1,2,1], 'val': [10,20,30]})
    right = pd.DataFrame({'yr': [2020,2020,2021], 'mon': [1,2,1], 'metric': [100,200,300]})
    # WRONG: only guesses 'year' -> 'yr', missing month
    result = left.merge(right, left_on='year', right_on='yr', how='inner', validate='one_to_many')
    return result
"""
    },

    "MERGE_004": {
        "description": "Genuine smell: Default inner join but business needs FULL OUTER join (e.g., union of two time series).",
        "reasoning": "Contract PBT infers row_count_invariant: gte_max (both sides preserved).",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_time_series():
    ts1 = pd.DataFrame({'date': ['2024-01-01','2024-01-02'], 'value1': [1,2]})
    ts2 = pd.DataFrame({'date': ['2024-01-02','2024-01-03'], 'value2': [3,4]})
    result = ts1.merge(ts2, on='date')
    return result
""",
        "after_correct": """
def merge_time_series():
    ts1 = pd.DataFrame({'date': ['2024-01-01','2024-01-02'], 'value1': [1,2]})
    ts2 = pd.DataFrame({'date': ['2024-01-02','2024-01-03'], 'value2': [3,4]})
    result = ts1.merge(ts2, on='date', how='outer', validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_time_series():
    ts1 = pd.DataFrame({'date': ['2024-01-01','2024-01-02'], 'value1': [1,2]})
    ts2 = pd.DataFrame({'date': ['2024-01-02','2024-01-03'], 'value2': [3,4]})
    # WRONG: inner join drops 2024-01-01 and 2024-01-03
    result = ts1.merge(ts2, on='date', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_005": {
        "description": "False Positive: Merge already has all explicit parameters correctly set.",
        "reasoning": "Stage1 should return NO_SMELL. No refactoring needed.",
        "smell_type": "merge",
        "expected_stage1": "NO_SMELL",
        "expected_validator": "NO_SMELL",
        "before": """
def explicit_merge():
    df1 = pd.DataFrame({'key': [1,2], 'val': ['a','b']})
    df2 = pd.DataFrame({'key': [1,3], 'val2': ['c','d']})
    result = df1.merge(df2, on='key', how='left', validate='one_to_one')
    return result
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "MERGE_006": {
        "description": "False Positive: Merge using left_index and right_index (special API).",
        "reasoning": "These are intentional index-based merges. Not a smell. Stage1 should recognize and skip.",
        "smell_type": "merge",
        "expected_stage1": "NO_SMELL",
        "expected_validator": "NO_SMELL",
        "before": """
def index_merge():
    df1 = pd.DataFrame({'val': [10,20]}, index=[1,2])
    df2 = pd.DataFrame({'val2': [100,200]}, index=[1,2])
    result = df1.merge(df2, left_index=True, right_index=True)
    return result
""",
        "after_correct": "",
        "after_wrong": "",
    },

    "MERGE_007": {
        "description": "Wrong refactoring: SLM hallucinates column name 'id' instead of actual 'customer_id'.",
        "reasoning": "Contract PBT recovers schema -> infers on_key='customer_id'. PBT generates mismatched keys -> fails.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def merge_hallucinated():
    users = pd.DataFrame({'user_key': [1,2,3], 'name': ['A','B','C']})
    orders = pd.DataFrame({'cust_key': [1,2], 'amt': [100,200]})
    result = users.merge(orders)
    return result
""",
        "after_correct": "",
        "after_wrong": """
def merge_hallucinated():
    users = pd.DataFrame({'user_key': [1,2,3], 'name': ['A','B','C']})
    orders = pd.DataFrame({'cust_key': [1,2], 'amt': [100,200]})
    # WRONG: Guesses 'id' which doesn't exist
    result = users.merge(orders, left_on='id', right_on='id', how='inner', validate='one_to_many')
    return result
"""
    },

    "MERGE_008": {
        "description": "Wrong refactoring: SLM chooses 'how='inner'' but contract infers 'outer' from surrounding logic.",
        "reasoning": "Context (e.g., filling missing dates) implies outer. Contract PBT catches how mismatch.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def fill_missing_dates():
    df_actual = pd.DataFrame({'date': ['2024-01-01','2024-01-03'], 'val': [1,3]})
    df_all = pd.DataFrame({'date': ['2024-01-01','2024-01-02','2024-01-03']})
    result = df_actual.merge(df_all, on='date')
    return result
""",
        "after_correct": "",
        "after_wrong": """
def fill_missing_dates():
    df_actual = pd.DataFrame({'date': ['2024-01-01','2024-01-03'], 'val': [1,3]})
    df_all = pd.DataFrame({'date': ['2024-01-01','2024-01-02','2024-01-03']})
    # WRONG: inner loses the missing date row
    result = df_actual.merge(df_all, on='date', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_009": {
        "description": "Wrong refactoring: Missing suffixes leads to column name collision.",
        "reasoning": "Contract PBT should verify that no ambiguous columns exist in the result. This catches missed suffixes.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def merge_overlap():
    left = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    right = pd.DataFrame({'id': [1,2], 'col': ['c','d']})
    result = left.merge(right, on='id')
    return result
""",
        "after_correct": """
def merge_overlap():
    left = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    right = pd.DataFrame({'id': [1,2], 'col': ['c','d']})
    result = left.merge(right, on='id', suffixes=('_left', '_right'))
    return result
""",
        "after_wrong": """
def merge_overlap():
    left = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    right = pd.DataFrame({'id': [1,2], 'col': ['c','d']})
    # WRONG: no suffixes -> columns conflict or overwritten
    result = left.merge(right, on='id', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_010": {
        "description": "Genuine smell: Merging on a column that needs type conversion (context indicates casting).",
        "reasoning": "Tests if SLM-2 can infer that 'id' needs to be converted to int/string.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_type_mismatch():
    users = pd.DataFrame({'uid': ['1','2','3'], 'name': ['A','B','C']})
    orders = pd.DataFrame({'user_id': [1,2], 'amt': [10,20]})
    result = users.merge(orders, left_on='uid', right_on='user_id')
    return result
""",
        "after_correct": """
def merge_type_mismatch():
    users = pd.DataFrame({'uid': ['1','2','3'], 'name': ['A','B','C']})
    orders = pd.DataFrame({'user_id': [1,2], 'amt': [10,20]})
    # Correct: explicit cast in context? Actually, pandas handles mixed types via object.
    result = users.merge(orders, left_on='uid', right_on='user_id', how='inner', validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_type_mismatch():
    users = pd.DataFrame({'uid': ['1','2','3'], 'name': ['A','B','C']})
    orders = pd.DataFrame({'user_id': [1,2], 'amt': [10,20]})
    # WRONG: guesses on='id' when neither side has that column
    result = users.merge(orders, left_on='id', right_on='id', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_011": {
        "description": "Genuine smell: Merge with duplicate column names (requires suffixes).",
        "reasoning": "SLM-1 should add suffixes. Contract PBT checks column uniqueness.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_duplicate_cols():
    df1 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    df2 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    result = df1.merge(df2, on='key')
    return result
""",
        "after_correct": """
def merge_duplicate_cols():
    df1 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    df2 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    result = df1.merge(df2, on='key', suffixes=('_x', '_y'), validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_duplicate_cols():
    df1 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    df2 = pd.DataFrame({'key': [1,2], 'date': ['2020-01-01','2020-01-02']})
    result = df1.merge(df2, on='key', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_012": {
    "description": "Genuine smell: Merge with 'how' explicitly set, but 'on' is implicit (single common column). "
                   "HARP‑ML requires explicit 'on' (or left_on/right_on) for clarity.",
    "reasoning": "The smell definition flags missing 'on' – even if there's only one common column, explicit 'on' is required.",
    "smell_type": "merge",
    "expected_stage1": "GENUINE",      # Changed from NO_SMELL
    "expected_validator": "VALIDATED", # Assuming correct refactoring would add on and validate
    "before": """
def implicit_on_merge():
    df1 = pd.DataFrame({'id': [1,2], 'val': [10,20]})
    df2 = pd.DataFrame({'id': [1,3], 'val2': [100,200]})
    result = df1.merge(df2, how='inner')
    return result
""",
    "after_correct": """
def implicit_on_merge_fixed():
    df1 = pd.DataFrame({'id': [1,2], 'val': [10,20]})
    df2 = pd.DataFrame({'id': [1,3], 'val2': [100,200]})
    result = df1.merge(df2, on='id', how='inner', validate='one_to_one')
    return result
""",
    "after_wrong": """
def implicit_on_merge_wrong():
    df1 = pd.DataFrame({'id': [1,2], 'val': [10,20]})
    df2 = pd.DataFrame({'id': [1,3], 'val2': [100,200]})
    result = df1.merge(df2, left_on='wrong', right_on='wrong', how='inner')
    return result
"""
},

    "MERGE_013": {
        "description": "Genuine smell: Merging a large DataFrame with itself (self-join) missing validate.",
        "reasoning": "Self-joins often need validate='many_to_one' to avoid accidental Cartesian products.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def self_join_employees():
    employees = pd.DataFrame({'emp_id': [1,2,3,4], 'mgr_id': [None,1,1,2], 'name': ['A','B','C','D']})
    result = employees.merge(employees, left_on='mgr_id', right_on='emp_id')
    return result
""",
        "after_correct": """
def self_join_employees():
    employees = pd.DataFrame({'emp_id': [1,2,3,4], 'mgr_id': [None,1,1,2], 'name': ['A','B','C','D']})
    result = employees.merge(employees, left_on='mgr_id', right_on='emp_id', how='left', suffixes=('_emp', '_mgr'), validate='many_to_one')
    return result
""",
        "after_wrong": """
def self_join_employees():
    employees = pd.DataFrame({'emp_id': [1,2,3,4], 'mgr_id': [None,1,1,2], 'name': ['A','B','C','D']})
    # WRONG: no suffixes, columns collide; validate missing
    result = employees.merge(employees, left_on='mgr_id', right_on='emp_id', how='left')
    return result
"""
    },

    "MERGE_014": {
        "description": "Wrong refactoring: SLM guesses 'validate='one_to_one'' but duplicates exist in right table.",
        "reasoning": "Contract PBT will generate data with duplicate keys, causing MergeError. Catches validate mis-spec.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def merge_with_dups():
    left = pd.DataFrame({'key': [1,2], 'val': ['a','b']})
    right = pd.DataFrame({'key': [1,1,2], 'val2': ['x','y','z']})
    result = left.merge(right, on='key')
    return result
""",
        "after_correct": "",
        "after_wrong": """
def merge_with_dups():
    left = pd.DataFrame({'key': [1,2], 'val': ['a','b']})
    right = pd.DataFrame({'key': [1,1,2], 'val2': ['x','y','z']})
    # WRONG: one_to_one will crash if duplicates exist
    result = left.merge(right, on='key', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_015": {
        "description": "Genuine smell: Merging with 'on' parameter missing despite common column named differently.",
        "reasoning": "Standard context where left_on and right_on are required.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_asymmetric_keys():
    table_a = pd.DataFrame({'order_id': [101,102], 'qty': [1,2]})
    table_b = pd.DataFrame({'id': [101,103], 'status': ['shipped','pending']})
    result = table_a.merge(table_b)
    return result
""",
        "after_correct": """
def merge_asymmetric_keys():
    table_a = pd.DataFrame({'order_id': [101,102], 'qty': [1,2]})
    table_b = pd.DataFrame({'id': [101,103], 'status': ['shipped','pending']})
    result = table_a.merge(table_b, left_on='order_id', right_on='id', how='inner', validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_asymmetric_keys():
    table_a = pd.DataFrame({'order_id': [101,102], 'qty': [1,2]})
    table_b = pd.DataFrame({'id': [101,103], 'status': ['shipped','pending']})
    # WRONG: guesses 'on' as 'id' which doesn't exist in left
    result = table_a.merge(table_b, on='id', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_016": {
        "description": "Wrong refactoring: Adds 'validate' but uses incorrect 'how' causing data loss.",
        "reasoning": "Contract PBT catches row count invariant mismatch.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def merge_preserve_all():
    left = pd.DataFrame({'id': [1,2,3], 'val': ['a','b','c']})
    right = pd.DataFrame({'id': [1,2], 'val2': ['x','y']})
    result = left.merge(right, on='id')
    return result
""",
        "after_correct": """
def merge_preserve_all():
    left = pd.DataFrame({'id': [1,2,3], 'val': ['a','b','c']})
    right = pd.DataFrame({'id': [1,2], 'val2': ['x','y']})
    result = left.merge(right, on='id', how='left', validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_preserve_all():
    left = pd.DataFrame({'id': [1,2,3], 'val': ['a','b','c']})
    right = pd.DataFrame({'id': [1,2], 'val2': ['x','y']})
    # WRONG: inner drops row 3
    result = left.merge(right, on='id', how='inner', validate='one_to_one')
    return result
"""
    },

    "MERGE_017": {
    "description": "Genuine smell: Merge with 'validate' present but 'how' implicit (pandas default inner). "
                   "HARP‑ML requires explicit 'how' even if default is inner.",
    "reasoning": "Structure is partially explicit, but missing 'how' – so it's a smell.",
    "smell_type": "merge",
    "expected_stage1": "GENUINE",      # Changed from NO_SMELL
    "expected_validator": "VALIDATED",
    "before": """
def merge_validate_only():
    df1 = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    df2 = pd.DataFrame({'id': [1,2], 'col2': ['c','d']})
    result = df1.merge(df2, on='id', validate='one_to_one')
    return result
""",
    "after_correct": """
def merge_validate_only_fixed():
    df1 = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    df2 = pd.DataFrame({'id': [1,2], 'col2': ['c','d']})
    result = df1.merge(df2, on='id', how='inner', validate='one_to_one')
    return result
""",
    "after_wrong": """
def merge_validate_only_wrong():
    df1 = pd.DataFrame({'id': [1,2], 'col': ['a','b']})
    df2 = pd.DataFrame({'id': [1,2], 'col2': ['c','d']})
    result = df1.merge(df2, left_on='id', right_on='wrong', how='inner')
    return result
"""
},

    "MERGE_018": {
        "description": "Genuine smell: Merging with indicator flag (to track left/right) but missing core params.",
        "reasoning": "Indicator is supplementary; missing on/how is still a smell.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_with_indicator():
    left = pd.DataFrame({'k': [1,2], 'v': ['a','b']})
    right = pd.DataFrame({'k': [1,3], 'v2': ['c','d']})
    result = left.merge(right, indicator=True)
    return result
""",
        "after_correct": """
def merge_with_indicator():
    left = pd.DataFrame({'k': [1,2], 'v': ['a','b']})
    right = pd.DataFrame({'k': [1,3], 'v2': ['c','d']})
    result = left.merge(right, on='k', how='outer', indicator=True, validate='one_to_one')
    return result
""",
        "after_wrong": """
def merge_with_indicator():
    left = pd.DataFrame({'k': [1,2], 'v': ['a','b']})
    right = pd.DataFrame({'k': [1,3], 'v2': ['c','d']})
    # WRONG: guesses inner, loses row 2 from left and row 3 from right
    result = left.merge(right, on='k', how='inner', indicator=True, validate='one_to_one')
    return result
"""
    },

    "MERGE_019": {
        "description": "Wrong refactoring: SLM adds 'validate' but chooses wrong side ('one_to_many' instead of 'many_to_one').",
        "reasoning": "Contract PBT infers cardinality from data (e.g., many products per category). Catches reverse cardinality.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "REGRESSION",
        "before": """
def merge_cardinality():
    categories = pd.DataFrame({'cat_id': [1,2], 'cat_name': ['Electronics','Books']})
    products = pd.DataFrame({'category': [1,1,2], 'prod': ['TV','Radio','Novel']})
    result = categories.merge(products)
    return result
""",
        "after_correct": """
def merge_cardinality():
    categories = pd.DataFrame({'cat_id': [1,2], 'cat_name': ['Electronics','Books']})
    products = pd.DataFrame({'category': [1,1,2], 'prod': ['TV','Radio','Novel']})
    result = categories.merge(products, left_on='cat_id', right_on='category', how='inner', validate='one_to_many')
    return result
""",
        "after_wrong": """
def merge_cardinality():
    categories = pd.DataFrame({'cat_id': [1,2], 'cat_name': ['Electronics','Books']})
    products = pd.DataFrame({'category': [1,1,2], 'prod': ['TV','Radio','Novel']})
    # WRONG: validate='many_to_one' expects many cats per product — fails
    result = categories.merge(products, left_on='cat_id', right_on='category', how='inner', validate='many_to_one')
    return result
"""
    },

    "MERGE_020": {
        "description": "Edge case: Merging DataFrames with MultiIndex columns.",
        "reasoning": "Complex schema. Tests whether SLM-2 can handle multi-level column names.",
        "smell_type": "merge",
        "expected_stage1": "GENUINE",
        "expected_validator": "VALIDATED",
        "before": """
def merge_multiindex():
    cols1 = pd.MultiIndex.from_tuples([('A','x'), ('A','y')])
    cols2 = pd.MultiIndex.from_tuples([('B','x'), ('B','z')])
    df1 = pd.DataFrame([[1,2]], columns=cols1)
    df2 = pd.DataFrame([[1,3]], columns=cols2)
    result = df1.merge(df2)
    return result
""",
        "after_correct": """
def merge_multiindex():
    cols1 = pd.MultiIndex.from_tuples([('A','x'), ('A','y')])
    cols2 = pd.MultiIndex.from_tuples([('B','x'), ('B','z')])
    df1 = pd.DataFrame([[1,2]], columns=cols1)
    df2 = pd.DataFrame([[1,3]], columns=cols2)
    result = df1.merge(df2, left_on=[('A','x')], right_on=[('B','x')], how='inner')
    return result
""",
        "after_wrong": """
def merge_multiindex():
    cols1 = pd.MultiIndex.from_tuples([('A','x'), ('A','y')])
    cols2 = pd.MultiIndex.from_tuples([('B','x'), ('B','z')])
    df1 = pd.DataFrame([[1,2]], columns=cols1)
    df2 = pd.DataFrame([[1,3]], columns=cols2)
    # WRONG: flattens and guesses 'x' — ambiguous
    result = df1.merge(df2, on='x', how='inner', validate='one_to_one')
    return result
"""
    },
}