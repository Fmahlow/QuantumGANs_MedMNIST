# LaSt-QGAN

Realization of [Latent Style-based Quantum GAN for high-quality Image Generation](https://arxiv.org/abs/2406.02668) trained on **MNIST** using **PyTorch Lightning**

To start **Autoencoder** training:
```
python3 train_autoencoder.py --config autoencoder.yaml
```

To start **QGAN** training:
```
python3 train_qgan.py --config gan.yaml --autoencoder-config autoencoder.yaml
```

Examples of generated images:

![](images/Samples%20Generated_119_c036c864503b5cc7a1ce.png)
![](images/Samples%20Generated_239_d4e8a23173737aa852eb.png)
![](images/Samples%20Generated_279_52a2143b309770bb2358.png)
![](images/Samples%20Generated_279_9580711b83835f7f58c6.png)
![](images/Samples%20Generated_319_9c251d959dccfb5a7bbd.png)
![](images/Samples%20Generated_439_738779f309d3c3e7eccd.png)
![](images/Samples%20Generated_599_468fc19ff4abb9b71583.png)
![](images/Samples%20Generated_759_2a1c7877d8f562437f82.png)

