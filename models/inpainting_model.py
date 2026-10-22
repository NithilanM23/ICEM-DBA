import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .generator import ICEM_DBA_Net
from .discriminator import Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss, BoundarySeamLoss

class ICEM_DBA_InpaintingModel(nn.Module):
    """
    High-level Inpainting Model coordinator managing generator, discriminator,
    losses, optimizers, learning rate scheduling, mixed precision, and checkpointing.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.iteration = 0
        self.device = getattr(config, 'DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Checkpoint paths
        save_dir = getattr(config, 'PATH', './checkpoints')
        self.gen_weights_path = os.path.join(save_dir, 'icem_dba_gen.pth')
        self.dis_weights_path = os.path.join(save_dir, 'icem_dba_dis.pth')

        # Instantiate ICEM-DBA Generator & Discriminator
        dim = getattr(config, 'DIM', 48)
        num_blocks = getattr(config, 'NUM_BLOCKS', [4, 6, 6, 8])
        num_refinement_blocks = getattr(config, 'NUM_REFINEMENT_BLOCKS', 4)
        d_state = getattr(config, 'D_STATE', 16)
        inner_rank = getattr(config, 'INNER_RANK', 32)
        d_key = getattr(config, 'D_KEY', 16)

        self.generator = ICEM_DBA_Net(
            inp_channels=4,
            out_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            d_state=d_state,
            inner_rank=inner_rank,
            d_key=d_key,
            use_checkpoint=bool(getattr(config, 'USE_CHECKPOINT', True)),
            patch_size=getattr(config, 'PATCH_SIZE', 4)
        ).to(self.device)

        use_sigmoid = (getattr(config, 'GAN_LOSS', 'lsgan') != 'hinge')
        self.discriminator = Discriminator(
            in_channels=3,
            use_sigmoid=use_sigmoid
        ).to(self.device)

        # Multi-GPU support if configured
        gpus = getattr(config, 'GPU', [0])
        use_checkpoint = bool(getattr(config, 'USE_CHECKPOINT', True))
        if torch.cuda.is_available() and len(gpus) > 1 and torch.cuda.device_count() > 1:
            if use_checkpoint:
                print(f"[Hardware] Notice: Activation checkpointing is active. PyTorch DataParallel multi-threading")
                print(f"           conflicts with autograd checkpointing and spectral norm cuBLAS contexts.")
                print(f"           Running stably on primary GPU [{gpus[0]}] with 48GB VRAM buffer.")
            else:
                print(f"[Model] Wrapping Generator & Discriminator with nn.DataParallel across GPUs: {gpus}")
                self.generator = nn.DataParallel(self.generator, device_ids=gpus)
                self.discriminator = nn.DataParallel(self.discriminator, device_ids=gpus)

        # Loss Criteria
        self.vgg_pretrained = getattr(config, 'VGG_PRETRAINED', True)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.perceptual_loss = PerceptualLoss(pretrained=self.vgg_pretrained).to(self.device)
        self.style_loss = StyleLoss(pretrained=self.vgg_pretrained).to(self.device)
        self.tv_loss = TVLoss().to(self.device)
        self.bsd_loss = BoundarySeamLoss().to(self.device)
        self.adversarial_loss = AdversarialLoss(type=getattr(config, 'GAN_LOSS', 'lsgan')).to(self.device)

        # Optimizers
        lr = float(getattr(config, 'LR', 1e-4))
        d2g_lr = float(getattr(config, 'D2G_LR', 0.1))
        beta1 = float(getattr(config, 'BETA1', 0.9))
        beta2 = float(getattr(config, 'BETA2', 0.999))

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=lr * d2g_lr,
            betas=(beta1, beta2)
        )

        # Schedulers
        milestones = getattr(config, 'MILESTONES', [20000, 40000, 60000, 80000, 120000])
        gamma = float(getattr(config, 'LR_DECAY', 0.5))
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(self.gen_optimizer, milestones=milestones, gamma=gamma)
        self.dis_scheduler = optim.lr_scheduler.MultiStepLR(self.dis_optimizer, milestones=milestones, gamma=gamma)

        # Mixed Precision Scaler (default disabled for robust FP32 stability)
        self.use_amp = bool(getattr(config, 'AMP', False)) and torch.cuda.is_available() and ('cuda' in str(self.device))
        self.device_type = 'cuda' if ('cuda' in str(self.device)) else 'cpu'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def forward(self, images, masks):
        return self.generator(images, masks)

    def process(self, images, masks):
        """
        Executes forward pass, evaluates generator and discriminator loss objectives,
        and executes clean 2-phase optimizer steps. Supports gradient accumulation.
        """
        accum_steps = max(1, int(getattr(self.config, 'GRAD_ACCUM_STEPS', 1)))
        self.accum_counter = getattr(self, 'accum_counter', 0) + 1
        is_accum_step = (self.accum_counter % accum_steps == 0)

        use_amp = getattr(self, 'use_amp', False)
        device_type = getattr(self, 'device_type', 'cuda' if ('cuda' in str(self.device)) else 'cpu')

        # -------------------------------------------------------------
        # 1. Generator forward pass (initial prediction)
        # -------------------------------------------------------------
        with torch.amp.autocast(device_type, enabled=use_amp):
            outputs_img = self(images, masks)

        # -------------------------------------------------------------
        # 2. Phase 1: Train Discriminator
        # -------------------------------------------------------------
        if (self.accum_counter - 1) % accum_steps == 0:
            self.dis_optimizer.zero_grad()

        with torch.amp.autocast(device_type, enabled=use_amp):
            dis_real, _ = self.discriminator(images)
            dis_fake, _ = self.discriminator(outputs_img.detach())

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2.0
            dis_loss_scaled = dis_loss / float(accum_steps)

        self.scaler.scale(dis_loss_scaled).backward()

        if is_accum_step:
            self.scaler.step(self.dis_optimizer)

        # -------------------------------------------------------------
        # 3. Phase 2: Train Generator
        # -------------------------------------------------------------
        if (self.accum_counter - 1) % accum_steps == 0:
            self.gen_optimizer.zero_grad()

        with torch.amp.autocast(device_type, enabled=use_amp):
            gen_fake, _ = self.discriminator(outputs_img)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * getattr(self.config, 'INPAINT_ADV_LOSS_WEIGHT', 0.01)

            mask_mean = torch.clamp(torch.mean(masks.float()), min=1e-4)
            gen_l1_loss = self.l1_loss(outputs_img, images) * getattr(self.config, 'L1_LOSS_WEIGHT', 1.0) / mask_mean

            if getattr(self, 'vgg_pretrained', True):
                gen_content_loss = self.perceptual_loss(outputs_img, images) * getattr(self.config, 'CONTENT_LOSS_WEIGHT', 0.1)
                gen_style_loss = self.style_loss(outputs_img * masks, images * masks) * getattr(self.config, 'STYLE_LOSS_WEIGHT', 250.0)
            else:
                gen_content_loss = torch.tensor(0.0, device=self.device)
                gen_style_loss = torch.tensor(0.0, device=self.device)

            gen_tv_loss = self.tv_loss(outputs_img) * getattr(self.config, 'TV_LOSS_WEIGHT', 0.0)
            gen_bsd_loss = self.bsd_loss(outputs_img, images, masks) * getattr(self.config, 'BSD_LOSS_WEIGHT', 1.0)

            gen_loss = gen_gan_loss + gen_l1_loss + gen_content_loss + gen_style_loss + gen_tv_loss + gen_bsd_loss
            gen_loss_scaled = gen_loss / float(accum_steps)

        self.scaler.scale(gen_loss_scaled).backward()

        if is_accum_step:
            self.iteration += 1
            scale_before = self.scaler.get_scale() if use_amp else 1.0
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale() if use_amp else 1.0

            # Step schedulers only if optimizer step was executed
            if not use_amp or scale_before <= scale_after:
                self.gen_scheduler.step()
                self.dis_scheduler.step()

        logs = [
            ("gLoss", gen_loss.item()),
            ("dLoss", dis_loss.item()),
            ("l1", gen_l1_loss.item()),
            ("style", gen_style_loss.item()),
            ("content", gen_content_loss.item()),
            ("bsd", gen_bsd_loss.item())
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    def backward(self, gen_loss=None, dis_loss=None):
        """
        API compatibility stub. Backpropagation and optimizer steps are executed
        directly inside process() in strict 2-phase sequence.
        """
        pass

    def save(self):
        """
        Saves generator and discriminator model weights.
        """
        save_dir = getattr(self.config, 'PATH', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[Checkpoint] Saving model at iteration {self.iteration}...")

        # Unwrap DataParallel before saving for universal compatibility (single/multi-GPU)
        gen_state = self.generator.module.state_dict() if isinstance(self.generator, nn.DataParallel) else self.generator.state_dict()
        dis_state = self.discriminator.module.state_dict() if isinstance(self.discriminator, nn.DataParallel) else self.discriminator.state_dict()

        torch.save({
            'iteration': self.iteration,
            'generator': gen_state,
            'gen_opt': self.gen_optimizer.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': dis_state,
            'dis_opt': self.dis_optimizer.state_dict()
        }, self.dis_weights_path)

        # Milestone snapshot every 5,000 iterations
        if self.iteration % 5000 == 0:
            milestone_path = os.path.join(save_dir, f'icem_dba_gen_{self.iteration:07d}.pth')
            torch.save({
                'iteration': self.iteration,
                'generator': gen_state,
            }, milestone_path)
            print(f"[Checkpoint] Saved milestone snapshot to {milestone_path}")

    def load(self):
        """
        Loads pre-trained model weights if checkpoint files exist.
        """
        target_gen = self.generator.module if isinstance(self.generator, nn.DataParallel) else self.generator
        target_dis = self.discriminator.module if isinstance(self.discriminator, nn.DataParallel) else self.discriminator

        if os.path.exists(self.gen_weights_path):
            print(f"[Checkpoint] Loading generator from {self.gen_weights_path}...")
            data = torch.load(self.gen_weights_path, map_location=self.device)
            gen_weights = data['generator']
            # Strip 'module.' prefix if present from previous DataParallel saves
            gen_weights = {k.replace('module.', ''): v for k, v in gen_weights.items()}
            target_gen.load_state_dict(gen_weights, strict=False)
            self.iteration = data.get('iteration', 0)
            print(f"[Checkpoint] Resuming from iteration {self.iteration}")

        if os.path.exists(self.dis_weights_path):
            print(f"[Checkpoint] Loading discriminator from {self.dis_weights_path}...")
            data = torch.load(self.dis_weights_path, map_location=self.device)
            dis_weights = data['discriminator']
            dis_weights = {k.replace('module.', ''): v for k, v in dis_weights.items()}
            target_dis.load_state_dict(dis_weights, strict=False)
