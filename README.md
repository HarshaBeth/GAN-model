# DC GAN Model - Handwritten Digit Generator

### Impact 
Generating realistic images has many useful impacts, such as creating immersive experiences in video games, creating realistic faces for animation, and much more! This project is a step forward to understand the process of  GANs. I used the MNIST dataset to train our model and with 10,000 epochs our GAN model quickly learned to generate realistic images of handwritten digits as seen below.

![image](https://github.com/user-attachments/assets/c2c00c1f-95e5-4165-8baa-8b438e0c8376)

# Generated Images of Handwritten Digits
![Generated Images](https://github.com/user-attachments/assets/7f55048b-dc3c-4266-9ca7-a55044f43006)


<hr>

Imported modules:
<li>Numpy</li>
<li>Tensorflow</li>
<li>Matplotlib</li>
<li>Os</li>

<br>

# Building Generator

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
<br>

# Building Discriminator

To build our discriminator we basically do the opposite of our generator. We achieve this by using the Flatten, Dense, and LeakyReLU Layers. 

We take the noise image generated by the generator and Flatten it for our discriminator to be able to "see" it. After our generator reads the image, it its last Dense layer it will have 1 nueron with an ```activation='sigmoid'```. This activation function will be binary as our discriminator will be deciding whether the image is 'real' or 'fake'.

```
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28,28,1)),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])

    return model
```
# Building Complete GAN

Lastly, we build our GAN as a whole, because within the GAN is where our **generator** will be trained to fool our discriminator. Whereas, our discriminator will be trained independently outside our GAN.

```
gan_model = Sequential([generator, discriminator])
gan_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
```

After this step, we load our MNIST dataset and normalize the data to be between -1 and 1, plus we add a channel dimension as the discriminator will require it. Now the shape of our dataset looks like 
'(60000, 28, 28, 1)'.


# Training GAN Model

We have reached our last step of the process, training our model! We choose random **real** images with the desired batch size from our complete training dataset. Next, we generate random noise and feed it to our generator to produce generated images. Moving forward, we train our discriminator with the real and fake images while giving them labels, this is how our discriminator will be trained independently.

We have trained our discriminator, so the next step is to train our generator through the GAN model. We generate new noise again and feed it to the generator inside the GAN. The generator's task now is to generate new images that the discriminator will classify it as 'valid'.

To ensure our progress, every 1,000 epochs I will save our generated images. Furthermore, as a safety measure, I checkpointed the weights of our generator and discriminator.

```
def train_gan(epochs, batch_size=128, save_interval=50, start_epoch=0):
    valid = np.ones((batch_size, 1))    # Real Label
    fake = np.zeros((batch_size, 1))    # Fake Label
    if not os.path.exists('gan_checkpoints'):
        os.makedirs('gan_checkpoints')

    for epoch in range(start_epoch, epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        print(f"Epoch: {epoch}")
        real_imgs = X_train[idx]

        # Generate fake images using our generator, we make 128 images each with 100 dimensions for noise
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        
        # Train the discriminator on batches of real and fake images
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan_model.train_on_batch(noise, valid)

        if epoch % save_interval == 0:
            print(f"{epoch} [D_loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]]")
            save_imgs(epoch, generator)
            # Save weights
            generator.save_weights(f'gan_checkpoints/generator_weights_epoch_{epoch}.weights.h5')
            discriminator.save_weights(f'gan_checkpoints/discriminator_weights_epoch_{epoch}.weights.h5')

```

```
train_gan(epochs=10001, batch_size=64, save_interval=1000)
```


