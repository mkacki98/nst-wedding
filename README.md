# Neural Style Transfer

Easy to run Neural Style Transfer for a given content and style images.

I used it create a wedding gift! 

# Background on the method

Neural Style Transfer [Gatys, 2015](https://arxiv.org/pdf/1508.06576.pdf) is a ML technique in which a network is trained to generate an output image `g` from a content image `c` (e.g. a photograph) which resembles a style image `s` (e.g. Picasso painting). 

Two prerequsities for a high-level understanding.

Firstly, VGG (Simonyan, 2015)[https://arxiv.org/abs/1409.1556] is a very large convolutional architecture trained on a benchmark image recognition task. As a result, the architecture learns how to pick up features of objects and can be used for transfer learning.

Secondly, last layers of convolutional architectures pick up high-level objects (like faces, buildings etc., the actual important bits in the picture). 

In [Gatys, 2015](https://arxiv.org/pdf/1508.06576.pdf), both `c` and `s` are passed through VGG to create internal representations of the images. The loss function of the architecture then combines two parts - content and style loss. 

Content loss measures how similar last layers of `g` and `c` are. The loss is 0 if they are exactly the same. 

On the other hand, style loss measures how similar all activations of `s` and `g` are once passed through VGG. 

You see these two losses are kind of pulling the way `g` will like in different directions (more like content or more like style). Since the final loss is a weighted combination of the two losses, by manipulating the weights you can shape the way the final image `g` will look like.

# Credits

Credits to [Alexis Jackq](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
