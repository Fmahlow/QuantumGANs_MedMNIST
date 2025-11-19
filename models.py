import torch.autograd as autograd
import torch.nn as nn
import torch


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()

        self.shape = [-1] + shape

    def forward(self, x):

        return x.view(self.shape)


class WassersteinLoss(nn.Module):

    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, real, fake):

        loss = -real + fake
        loss = loss.mean()

        return loss


class PenaltyLoss(nn.Module):

    def __init__(self, alpha):
        super(PenaltyLoss, self).__init__()

        self.alpha = alpha

    def forward(self, real, fake, discriminator):
        device = real.device
        batch_size = real.size(0)

        # Match the alpha tensor shape to the dimensionality of the inputs so that
        # interpolation preserves the original shape (e.g., 2D latent vectors
        # instead of unintentionally expanding to 4D tensors).
        alpha_shape = [batch_size] + [1] * (real.dim() - 1)
        alpha = torch.rand(alpha_shape, device=device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        logits = discriminator(interpolates)
        fake = torch.ones(logits.size(), device=device, requires_grad=False)

        gradients = autograd.grad(
            outputs=logits,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.alpha * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
