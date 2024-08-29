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

The input shape in the Dense layer is ```input_shape=(100,)``` because we allow our model to generate noise in a 100 different "ways", eventually the generate can turn this noise into a meaningful image. Furthermore, we used LeakyReLU to avoid the "dead neurons" issue. If many neurons in the network have negative values, this causes the gradient to become zero, which will limit the capacity of the network and stop learning. Thus, the LeakyReLU prevents the nuerons from dying.

Since we work with batches, we have to use BatchNormalization which will normalize our data during training and in batches instead of just normalizing once in the beginning. This helps act as a form of regularization.

Lastly, in the last Dense layer we have ```activation='tanh'``` because this activation function will scale the output of this layer to be between -1 and 1.

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


