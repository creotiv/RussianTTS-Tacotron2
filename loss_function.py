from torch import nn
import torch


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams, iteration=0):
        super(Tacotron2Loss, self).__init__()
        self.hparams = hparams
        self.guide_decay = 0.99999
        self.scale = 40.0 * (self.guide_decay**iteration)
        print('Guide scale:',self.scale)
        self.guide_lowbound = 1.0
        self.criterion_attention = nn.L1Loss()

    def forward(self, model_output, targets):
        _, mel_out, mel_out_postnet, gate_out, alignments_out, tpse_gst_pred,gst_target = model_output
        mel_target, gate_target, guide_target = targets[0], targets[1], targets[2]

        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        guide_target = guide_target.transpose(2,1)
        _,w,h = alignments_out.shape
        guide_target = guide_target[:,:w,:h]

        
        gate_out = gate_out.view(-1, 1)
        emb_loss = torch.tensor(0)
        if tpse_gst_pred is not None:
            emb_loss = nn.L1Loss()(tpse_gst_pred, gst_target.detach())
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = 1.3 * nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # loss_atten = torch.tensor(0)
        # if not self.hparams.no_dga:
        attention_masks = torch.ones_like(alignments_out)

        loss_atten = self.criterion_attention(
                guide_target * alignments_out * attention_masks,
                torch.zeros_like(alignments_out)) * self.scale
    
        self.scale *= self.guide_decay
        # self.scale = 100
        if self.scale < self.guide_lowbound:
            self.scale = self.guide_lowbound

        return mel_loss, gate_loss, loss_atten, emb_loss
