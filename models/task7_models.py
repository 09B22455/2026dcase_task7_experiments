from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
import torch.nn as nn
from .mcnn14 import MCnn14
from .vae import DomainCNNVAE
from .confidnet import ConfidNet
from config import D2_THRESHOLD, D3_THRESHOLD

class BaselineMelExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=1024,
            hop_length=320,
            win_length=1024,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=32000,
            n_fft=1024,
            n_mels=64,
            fmin=50,
            fmax=14000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True
        )

    def forward(self, waveform):
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)

        return x


# ==========================
# main model
# ==========================
class Task7Model(nn.Module):

    def __init__(self, task=1, d2_threshold = D2_THRESHOLD, d3_threshold = D3_THRESHOLD):

        super().__init__()
        self.task = task
        self.d2_threshold = d2_threshold
        self.d3_threshold = d3_threshold

        # ==========================
        # data
        # ==========================
        self.mel_extractor = BaselineMelExtractor()

        # ==========================
        # baseline classifier
        # ==========================
        self.classifier = MCnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=10,
            nb_tasks=3
        )

        # ==========================
        # VAE
        # ==========================
        self.vae_d2 = DomainCNNVAE(in_ch=1, latent_dim=128)
        if task == 2:
            self.vae_d3 = DomainCNNVAE(in_ch=1, latent_dim=128)

        # ==========================
        # ConfidNet
        # ==========================
        self.confid_d1 = ConfidNet(input_dim=2048)
        self.confid_d2 = ConfidNet(input_dim=2048)
        if task == 2:
            self.confid_d3 = ConfidNet(input_dim=2048)


    # ==========================
    # utils
    # ==========================
    def preprocess_for_vae(self, waveform):

        mel = self.mel_extractor(waveform)
        # (B,T,F)
        current_frames = mel.shape[2]

        if current_frames < 401:
            pad_len = 401 - current_frames
            pad = torch.zeros(
                mel.shape[0],
                mel.shape[1],
                pad_len,
                mel.shape[3],
                device=mel.device
            )
            mel = torch.cat([mel, pad], dim=2)
        else:
            mel = mel[:, :, :401, :]

        return mel


    def compute_kl(self, vae, waveform):
        mel = self.preprocess_for_vae(waveform)
        _, mu, logvar = vae(mel)

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )

        return kl

    # ==========================
    # forward
    # ==========================
    def forward_task1(self, waveform):

        kl_d2 = self.compute_kl(self.vae_d2, waveform)

        if kl_d2.item() > self.d2_threshold:
            return self.classifier(waveform, task=0)

        feat_d1 = self.classifier.extract_feature(waveform, task=0)
        feat_d2 = self.classifier.extract_feature(waveform, task=1)

        conf1 = self.confid_d1(feat_d1)
        conf2 = self.confid_d2(feat_d2)

        if conf1.item() >= conf2.item():
            selected = 0
        else:
            selected = 1

        return self.classifier(waveform, task=selected)
    
    def forward_task2(self, waveform):

        kl_d2 = self.compute_kl(self.vae_d2, waveform)

        kl_d3 = self.compute_kl(self.vae_d3, waveform)

        d2_reliable = (kl_d2.item()<=self.d2_threshold)
        d3_reliable = (kl_d3.item()<=self.d3_threshold)

        feat_d1 = self.classifier.extract_feature(waveform, task=0)
        feat_d2 = self.classifier.extract_feature(waveform, task=1)
        feat_d3 = self.classifier.extract_feature(waveform, task=2)

        conf1 = self.confid_d1(feat_d1).item()
        conf2 = self.confid_d2(feat_d2).item()
        conf3 = self.confid_d3(feat_d3).item()

        if (not d2_reliable) and (not d3_reliable):
            selected_task = 0

        elif (not d2_reliable) and d3_reliable:
            candidates = {
                0: conf1,
                2: conf3
            }
            selected_task = max(
                candidates,
                key=candidates.get
            )

        elif d2_reliable and (not d3_reliable):
            candidates = {
                0: conf1,
                1: conf2
            }
            selected_task = max(
                candidates,
                key=candidates.get
            )

        else:
            candidates = {
                0: conf1,
                1: conf2,
                2: conf3
            }
            selected_task = max(
                candidates,
                key=candidates.get
            )

        logits = self.classifier(waveform, task=selected_task)

        return logits
    
    def forward(self, waveform):

        if self.task == 1:
            return self.forward_task1(waveform)

        elif self.task == 2:
            return self.forward_task2(waveform)

        else:
            raise ValueError(f"Unsupported task={self.task}")


def load_model(model_path, task=1):
    model = Task7Model(task=task)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )
    model.eval()

    return model