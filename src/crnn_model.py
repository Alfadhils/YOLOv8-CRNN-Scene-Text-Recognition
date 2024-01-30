from torch import nn

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for sequence recognition.

    Args:
        img_channel (int): Number of input image channels.
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        num_class (int): Number of output classes.
        map_to_seq (int): Number of output features from CNN to be mapped to sequence.
        rnn_hidden (int): Number of hidden units in the LSTM layers.

    Attributes:
        cnn (nn.Sequential): Convolutional Neural Network backbone.
        map_to_seq (nn.Linear): Linear layer to map CNN output to sequence.
        rnn1 (nn.LSTM): First bidirectional LSTM layer.
        rnn2 (nn.LSTM): Second bidirectional LSTM layer.
        dense (nn.Linear): Fully connected layer for final classification.
    """

    def __init__(self, img_channel, img_height, img_width, num_class, map_to_seq=64, rnn_hidden=256):
        super(CRNN, self).__init__()

        # CNN Backbone
        self.cnn, (output_channel, output_height, output_width) = self._cnn_backbone(img_channel, img_height, img_width)

        # Mapping to Sequence
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq)

        # Bidirectional LSTM Layers
        self.rnn1 = nn.LSTM(map_to_seq, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        # Fully Connected Layer for Classification
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width):
        """
        Define the CNN backbone architecture.

        Args:
            img_channel (int): Number of input image channels.
            img_height (int): Height of input images.
            img_width (int): Width of input images.

        Returns:
            nn.Sequential: CNN backbone model.
            Tuple[int, int, int]: Output channel, height, and width after CNN layers.
        """
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            input_channel = channels[i]
            output_channel = channels[i + 1]

            # Convolutional layer
            cnn.add_module(f'conv{i}', nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i]))

            if batch_norm:
                # Batch Normalization
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            # ReLU activation
            relu = nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # CNN layers
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))

        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(6)
        output_channel, output_height, output_width = channels[-1], img_height // 16 - 1, img_width // 4 - 1

        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        """
        Forward pass of the CRNN model.

        Args:
            images (torch.Tensor): Input images with shape (batch, channel, height, width).

        Returns:
            torch.Tensor: Output logits with shape (seq_len, batch, num_class).
        """
        # CNN Backbone
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        # Reshape CNN output
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        # Bidirectional LSTM layers
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        # Fully Connected Layer for Classification
        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)
