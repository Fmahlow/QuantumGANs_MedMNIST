# QuantumGANs_MedMNIST

Este repositório contém exemplos de notebooks para treinamento de GANs clássicas e quânticas utilizando o dataset MedMNIST.

- **Requisitos principais**: Python 3.9+ (necessário para versões recentes do PennyLane que usam anotações de tipo como
  ``list[int]``). Usar Python 3.8 ou inferior causará erros de importação ao carregar o módulo ``pennylane``.

- `gan_classical_medmnist.ipynb`: implementa DCGAN, CGAN e WGAN-GP com avaliação por FID e Inception Score.
- `gan_classical_mendeley.ipynb`: versão adaptada para o dataset jxwvdwhpc2 disponível no Mendeley Data.
- `gan_quantum_medmnist.ipynb`: demonstra uma abordagem de Quantum GAN baseada no método de patches do Pennylane.
