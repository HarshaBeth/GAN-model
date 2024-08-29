# DC GAN Model - Handwritten Digit Generator

### Impact 
Generating realistic images has many useful impacts, such as creating immersive experiences in video games, creating realistic faces for animation, and much more! This project is a step forward to understand the process of  GANs. I used the MNIST dataset to train our model and with 10,000 epochs our GAN model quickly learned to generate realistic images of handwritten digits as seen below.

![image](https://github.com/user-attachments/assets/c2c00c1f-95e5-4165-8baa-8b438e0c8376)

<hr>

Imported modules:
<li>Numpy</li>
<li>Tensorflow</li>
<li>Matplotlib</li>
<li>Os</li>

<br>

The first step of building the GAN, is to have a generator that is able to generate random noise vectors and eventually learn to generate more realistic images that can fool the discriminator. Hence, to build our 
generator I used Dense, LeakyReLU, BatchNormalization, and Reshape Layers.

```
def build_generator():
    model = Sequential([
        Dense(256, input_shape=(100,)),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(28 * 28 * 1, activation='tanh'),
        Reshape((28, 28, 1))
    ])

    return model
```
